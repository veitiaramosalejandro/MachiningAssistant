"""Consume an RTSP audio stream and classify complete five-second windows.

RTSP is the input transport. FFmpeg performs decode/resampling and writes raw
PCM16LE to stdout; StreamingClassifier owns the five-second windowing and
never classifies an incomplete tail.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Integration.grpc_service import _load_env_file
from Integration.simulate_monitor import load_engine
from classification_streaming import StreamingClassifier


SAMPLE_RATE = 44100
CHANNELS = 2
FRAME_BYTES = CHANNELS * 2


def _env(env: dict[str, str], name: str, fallback: str) -> str:
    return env.get(name, fallback)


def _ffmpeg_command(ffmpeg: str, url: str, transport: str) -> list[str]:
    return [
        ffmpeg, "-hide_banner", "-loglevel", "warning",
        "-rtsp_transport", transport, "-i", url,
        "-vn", "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS), "pipe:1",
    ]


def _safe_rtsp_url(url: str) -> str:
    """Return an RTSP URL safe for logs and prediction artifacts."""
    try:
        parsed = urlsplit(url)
        if parsed.scheme.lower() != "rtsp" or not parsed.hostname:
            return "rtsp://<redacted>"
        netloc = parsed.hostname
        if parsed.port:
            netloc += f":{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
    except ValueError:
        return "rtsp://<redacted>"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _probe_stream(ffprobe: str, url: str, transport: str) -> dict:
    command = [
        ffprobe, "-v", "error", "-rtsp_transport", transport,
        "-select_streams", "a:0", "-show_streams", "-of", "json", url,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=15)
    if completed.returncode != 0:
        detail = completed.stderr.strip().replace(url, _safe_rtsp_url(url)) or f"exit code {completed.returncode}"
        raise RuntimeError(f"FFprobe could not inspect the RTSP stream: {detail}")
    try:
        payload = json.loads(completed.stdout)
        stream = (payload.get("streams") or [None])[0]
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError("FFprobe returned invalid stream metadata") from exc
    if not isinstance(stream, dict):
        raise RuntimeError("RTSP stream does not expose an audio track")
    fields = (
        "index", "codec_name", "codec_long_name", "profile", "sample_rate",
        "channels", "channel_layout", "sample_fmt", "bit_rate", "time_base",
        "start_time", "duration",
    )
    return {key: stream[key] for key in fields if key in stream}


def _write_metadata(path: Path, metadata: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def run_stream(*, model: str, checkpoint: Path, url: str, ffmpeg: str,
               ffprobe: str, transport: str, source_sample_rate: int, chunk_ms: float,
               output: Path, metadata_output: Path, device: str, threads: int) -> dict:
    if chunk_ms <= 0:
        raise ValueError("MACHINING_RTSP_CHUNK_MS must be positive")
    chunk_frames = round(SAMPLE_RATE * chunk_ms / 1000)
    if chunk_frames < 1:
        raise ValueError("RTSP chunk duration is shorter than one audio frame")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    safe_url = _safe_rtsp_url(url)
    source_probe = _probe_stream(ffprobe, url, transport)
    actual_source_rate = int(source_probe.get("sample_rate", source_sample_rate))
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    metadata = {
        "schema_version": 1,
        "session_id": session_id,
        "status": "starting",
        "started_at_utc": _utc_now(),
        "ended_at_utc": None,
        "transport": "rtsp",
        "rtsp_url": safe_url,
        "rtsp_transport": transport,
        "source_stream": source_probe,
        "source_sample_rate_hz": actual_source_rate,
        "model_sample_rate_hz": SAMPLE_RATE,
        "model_channels": CHANNELS,
        "conversion": {"codec": "pcm_s16le", "format": "s16le", "sample_rate_hz": SAMPLE_RATE, "channels": CHANNELS},
        "chunk_duration_ms": chunk_ms,
        "chunk_frames": chunk_frames,
        "checkpoint": str(checkpoint),
        "model": model,
        "chunks_received": 0,
        "source_frames_received": 0,
        "model_frames_received": 0,
        "processed_frames": 0,
        "emitted_windows": 0,
        "trailing_frames": 0,
        "ffmpeg_return_code": None,
        "error": None,
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    _write_metadata(metadata_output, metadata)

    import torch
    torch.set_num_threads(threads)
    predictor = load_engine(model, checkpoint, device=device)
    if predictor.sample_rate != SAMPLE_RATE or predictor.valid_frames != 220500:
        raise ValueError("Checkpoint must support 44100 Hz and five-second windows")
    stream = StreamingClassifier(predictor)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = _ffmpeg_command(ffmpeg, url, transport)
    started = time.perf_counter()
    emitted = 0
    pending = b""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    metadata["status"] = "running"
    metadata["ffmpeg_pid"] = process.pid
    _write_metadata(metadata_output, metadata)
    try:
        with output.open("w", encoding="utf-8") as handle:
            while True:
                raw = process.stdout.read(chunk_frames * FRAME_BYTES)
                if not raw:
                    break
                pending += raw
                complete_bytes = len(pending) - (len(pending) % FRAME_BYTES)
                if not complete_bytes:
                    continue
                payload, pending = pending[:complete_bytes], pending[complete_bytes:]
                pcm = np.frombuffer(payload, dtype="<i2").reshape(-1, CHANNELS)
                metadata["chunks_received"] += 1
                metadata["source_frames_received"] += len(pcm)
                metadata["model_frames_received"] = stream.input_frames + len(pcm)
                for result in stream.push(pcm, sample_rate=SAMPLE_RATE):
                    metadata["processed_frames"] = stream.processed_frames
                    metadata["emitted_windows"] = stream.emitted_windows
                    metadata["trailing_frames"] = stream.pending_frames
                    _write_metadata(metadata_output, metadata)
                    row = {
                        **result,
                        "transport": "rtsp",
                        "rtsp_url": safe_url,
                        "source_sample_rate_hz": actual_source_rate,
                        "model_sample_rate_hz": SAMPLE_RATE,
                        "metadata_file": str(metadata_output),
                        "stream_metadata": {
                            "session_id": session_id,
                            "chunks_received": metadata["chunks_received"],
                            "source_frames_received": metadata["source_frames_received"],
                            "model_frames_received": metadata["model_frames_received"],
                            "processed_frames": metadata["processed_frames"],
                        },
                        "emitted_after_start_s": time.perf_counter() - started,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    emitted += 1
                    print(f"RTSP result [{result['start_s']:.3f}, {result['end_s']:.3f}) s: "
                          f"{result['predicted_label']} ({result['confidence']:.6f})", flush=True)
        if pending:
            raise RuntimeError("FFmpeg ended with an incomplete PCM frame")
        return_code = process.wait(timeout=5)
        if return_code != 0:
            raise RuntimeError(f"FFmpeg could not read the RTSP stream; exit code {return_code}")
        tail = stream.finish()
        metadata.update({"status": "completed", "ended_at_utc": _utc_now(),
                         "processed_frames": stream.processed_frames,
                         "emitted_windows": stream.emitted_windows,
                         "trailing_frames": stream.pending_frames,
                         "ffmpeg_return_code": return_code})
        _write_metadata(metadata_output, metadata)
        return {"transport": "rtsp", "url": safe_url, "output": str(output),
                "metadata_output": str(metadata_output), "emitted_windows": emitted,
                "tail": tail, "ffmpeg_return_code": return_code}
    except KeyboardInterrupt:
        tail = stream.finish()
        metadata.update({"status": "interrupted", "ended_at_utc": _utc_now(),
                         "processed_frames": stream.processed_frames,
                         "emitted_windows": stream.emitted_windows,
                         "trailing_frames": stream.pending_frames})
        _write_metadata(metadata_output, metadata)
        return {"transport": "rtsp", "url": safe_url, "output": str(output),
                "metadata_output": str(metadata_output), "emitted_windows": emitted,
                "tail": tail, "ffmpeg_return_code": None, "status": "interrupted"}
    except Exception as exc:
        metadata.update({"status": "error", "ended_at_utc": _utc_now(), "error": str(exc),
                         "processed_frames": stream.processed_frames,
                         "emitted_windows": stream.emitted_windows,
                         "trailing_frames": stream.pending_frames})
        _write_metadata(metadata_output, metadata)
        raise
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        if process.stderr:
            error = process.stderr.read().decode("utf-8", errors="replace").strip()
            if error:
                print(f"FFmpeg: {error}", file=sys.stderr, flush=True)


def main(argv=None) -> int:
    env = _load_env_file()
    parser = argparse.ArgumentParser(description="Classify live audio from an RTSP stream")
    parser.add_argument("--url", default=_env(env, "MACHINING_RTSP_URL", ""))
    parser.add_argument("--transport", choices=("tcp", "udp"), default=_env(env, "MACHINING_RTSP_TRANSPORT", "tcp"))
    parser.add_argument("--ffmpeg", default=_env(env, "MACHINING_RTSP_FFMPEG", "ffmpeg"))
    parser.add_argument("--ffprobe", default=_env(env, "MACHINING_RTSP_FFPROBE", "ffprobe"))
    parser.add_argument("--source-sample-rate", type=int, default=int(_env(env, "MACHINING_RTSP_SOURCE_SAMPLE_RATE", "48000")))
    parser.add_argument("--chunk-ms", type=float, default=float(_env(env, "MACHINING_RTSP_CHUNK_MS", "20")))
    parser.add_argument("--output", type=Path, default=Path(_env(env, "MACHINING_RTSP_OUTPUT", "runs/rtsp/live_predictions.jsonl")))
    parser.add_argument("--metadata-output", type=Path,
                        default=Path(_env(env, "MACHINING_RTSP_METADATA_OUTPUT", "runs/rtsp/live_predictions.metadata.json")))
    parser.add_argument("--model", choices=("cnn", "lstm"), default=_env(env, "MACHINING_MODEL", "cnn"))
    parser.add_argument("--checkpoint", type=Path, default=Path(_env(env, "MACHINING_CHECKPOINT", "")))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=_env(env, "MACHINING_DEVICE", "auto"))
    parser.add_argument("--threads", type=int, default=int(_env(env, "MACHINING_THREADS", "2")))
    args = parser.parse_args(argv)
    if not args.url:
        parser.error("Set MACHINING_RTSP_URL in .env or pass --url")
    if not str(args.checkpoint):
        parser.error("Set MACHINING_CHECKPOINT in .env or pass --checkpoint")
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        report = run_stream(model=args.model, checkpoint=args.checkpoint.expanduser().resolve(),
                            url=args.url, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe, transport=args.transport,
                            source_sample_rate=args.source_sample_rate, chunk_ms=args.chunk_ms,
                            output=args.output.expanduser().resolve(),
                            metadata_output=args.metadata_output.expanduser().resolve(),
                            device=args.device, threads=args.threads)
    except (OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
