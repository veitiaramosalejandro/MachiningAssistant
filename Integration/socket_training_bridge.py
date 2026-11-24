"""Persist socket audio and CNC context as aligned RTSP training windows.

The socket transports context JSON and the Analyzer-side RTSP worker transports
PCM audio. This bridge joins both streams by ``sessionId`` and audio frame
position. It deliberately keeps raw context separate from the model context
contract and only emits a training context when all seven ``rtsp_stream``
features are present and numeric.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import tempfile
import time
import wave
from typing import Any

from Integration.rtsp_cnc_metadata import STREAM_FEATURE_NAMES, build_stream_feature_vector


SAMPLE_RATE_HZ = 44100
CHANNELS = 2
SAMPLE_BYTES = 2
WINDOW_FRAMES = SAMPLE_RATE_HZ * 5
WINDOW_BYTES = WINDOW_FRAMES * CHANNELS * SAMPLE_BYTES
MIN_TRAINING_SECONDS = 600
MINIMUM_LABELED_WINDOWS = MIN_TRAINING_SECONDS // 5
MODEL_CLASS_NAMES = ("excellent", "good", "sufficient", "bad", "very_bad", "broken")


def _checkpoint_contract() -> dict[str, Any]:
    return {
        "model_family": "CNN1D-VAE",
        "task": "classification",
        "context_contract": "rtsp_stream",
        "context_dim": len(STREAM_FEATURE_NAMES),
        "context_features": list(STREAM_FEATURE_NAMES),
        "class_names": list(MODEL_CLASS_NAMES),
    }


def _json_line(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def _stream_context_from_socket_context(raw: object, *, session_id: str, sequence: int) -> dict[str, Any]:
    """Adapt the explicit MachiningContext fields to the RTSP feature schema.

    No values are inferred. Missing fields remain missing and are rejected by
    ``build_stream_feature_vector`` below.
    """
    if not isinstance(raw, dict):
        raise ValueError("socket Context must be an object")

    def field(name: str) -> Any:
        return raw.get(name, raw.get(name[:1].lower() + name[1:]))

    # A future sender may already provide the independent stream contract.
    if isinstance(raw.get("context"), dict) or isinstance(raw.get("parameters"), dict):
        candidate = dict(raw)
        candidate.setdefault("type", "context")
        candidate.setdefault("recordingId", field("ExecutionId") or session_id)
        candidate.setdefault("sequence", sequence)
        return candidate

    candidate = {
        "type": "context",
        "recordingId": field("ExecutionId") or session_id,
        "sequence": sequence,
        "timestampSeconds": field("OccurredAtSeconds"),
        "context": {
            "context": {
                "Tool": {"DiameterMm": field("ToolDiameterMm")},
                "Parameters": {
                    "SpindleSpeed": {
                        "ProgrammedRpm": field("ProgrammedSpindleRpm"),
                        "AdjustmentPercent": field("SpindleOverridePercent"),
                        "AdjustedRpm": field("AdjustedSpindleRpm"),
                    },
                    "FeedRate": {
                        "ProgrammedMmPerMin": field("ProgrammedFeedMmPerMin"),
                        "AdjustmentPercent": field("FeedOverridePercent"),
                        "AdjustedMmPerMin": field("AdjustedFeedMmPerMin"),
                    },
                },
            }
        },
    }
    return candidate


class SocketTrainingBridge:
    """One persisted capture/training session per Analyzer socket session."""

    def __init__(self, root: Path, logger, *, transport: str = "tcp") -> None:
        self.root = root.expanduser().resolve()
        self.logger = logger
        self.transport = transport
        self._lock = threading.RLock()
        self._sessions: dict[str, dict[str, Any]] = {}

    def on_start(self, payload: dict[str, Any]) -> str:
        session_id = str(payload["sessionId"])
        with self._lock:
            if session_id in self._sessions:
                return "accepted"
            session_dir = self.root / session_id
            session_dir.mkdir(parents=True, exist_ok=False)
            job = {
                "listener_id": session_id,
                "session_id": session_id,
                "status": "starting",
                "pid": os.getpid(),
                "origin": payload.get("origin"),
                "machine_id": payload.get("machineId"),
                "execution_id": payload.get("executionId"),
                "rtsp_url": self._redact_url(str(payload["url"])),
                "transport": self.transport,
                "model_family": "CNN1D-VAE",
                "training_policy": "continuous_capture_then_batch",
                "min_training_seconds": MIN_TRAINING_SECONDS,
                "minimum_labeled_windows": MINIMUM_LABELED_WINDOWS,
                "training_ready": False,
                "labeled_windows": 0,
                "checkpoint_contract": _checkpoint_contract(),
                "session_dir": str(session_dir),
                "output": str(session_dir / "live_predictions.jsonl"),
                "metadata_output": str(session_dir / "session.metadata.json"),
                "log": str(session_dir / "listener.log"),
                "metadata_stream_output": str(session_dir / "stream_metadata.jsonl"),
                "metadata_status_output": str(session_dir / "metadata.status.json"),
                "metadata_log": str(session_dir / "metadata.listener.log"),
                "audio_windows_output": str(session_dir / "audio_windows"),
                "aligned_training_output": str(session_dir / "aligned_training.jsonl"),
                "socket_data_output": str(session_dir / "socket_data.jsonl"),
                "context_output": str(session_dir / "contexts.jsonl"),
                "latest_context": None,
                "latest_context_error": None,
                "audio_frames": 0,
                "audio_chunks": 0,
                "audio_bytes_received": 0,
                "audio_windows": 0,
                "last_sequence": 0,
                "started_at": time.time(),
                "metadata_status": "starting",
                "metadata_return_code": None,
                "observed_windows": 0,
                "labeled_duration_seconds": 0,
                "metadata_messages": 0,
                "metadata_events": 0,
                "quality_labels": [],
                "quality_label_scores": {},
                "metadata_snapshots": 0,
                "pending_audio": bytearray(),
            }
            Path(job["audio_windows_output"]).mkdir(parents=True, exist_ok=True)
            self._sessions[session_id] = job
            self._write_status(job)
        return "accepted"

    def on_data(self, payload: dict[str, Any]) -> str:
        session_id = str(payload["sessionId"])
        with self._lock:
            job = self._sessions.get(session_id)
            if job is None:
                raise ValueError("socket training session does not exist")
            _json_line(Path(job["socket_data_output"]), payload)
            job["last_sequence"] = int(payload["sequence"])
            job["metadata_messages"] = int(job["metadata_messages"]) + 1
            data = payload.get("data")
            raw_context = data.get("Context", data.get("context")) if isinstance(data, dict) else None
            if isinstance(raw_context, dict):
                try:
                    context = _stream_context_from_socket_context(
                        raw_context, session_id=session_id, sequence=int(payload["sequence"])
                    )
                    build_stream_feature_vector(context)
                    job["latest_context"] = context
                    job["latest_context_error"] = None
                    _json_line(Path(job["context_output"]), context)
                except (TypeError, ValueError) as exc:
                    job["latest_context"] = None
                    job["latest_context_error"] = str(exc)
            job["status"] = "receiving"
            job["metadata_status"] = "running"
            self._write_status(job)
        return "dataAccepted"

    def on_audio(self, chunk: dict[str, Any]) -> None:
        session_id = str(chunk["sessionId"])
        pcm = chunk.get("pcm16le")
        if not isinstance(pcm, (bytes, bytearray, memoryview)):
            raise ValueError("socket audio chunk must contain pcm16le bytes")
        with self._lock:
            job = self._sessions.get(session_id)
            if job is None:
                return
            pending: bytearray = job["pending_audio"]
            pending.extend(bytes(pcm))
            job["audio_chunks"] = int(job["audio_chunks"]) + 1
            job["audio_bytes_received"] = int(job["audio_bytes_received"]) + len(pcm)
            while len(pending) >= WINDOW_BYTES:
                payload = bytes(pending[:WINDOW_BYTES])
                del pending[:WINDOW_BYTES]
                index = int(job["audio_windows"])
                start_frame = index * WINDOW_FRAMES
                start_seconds = start_frame / SAMPLE_RATE_HZ
                end_seconds = start_seconds + 5.0
                wav_path = Path(job["audio_windows_output"]) / f"window_{index:06d}_{start_seconds:.3f}_{end_seconds:.3f}.wav"
                with wave.open(str(wav_path), "wb") as writer:
                    writer.setnchannels(CHANNELS)
                    writer.setsampwidth(SAMPLE_BYTES)
                    writer.setframerate(SAMPLE_RATE_HZ)
                    writer.writeframes(payload)
                context = job.get("latest_context")
                row = {
                    "window_start_seconds": start_seconds,
                    "window_end_seconds": end_seconds,
                    "audio_wav_file": str(Path("audio_windows") / wav_path.name),
                    "context": context,
                    "context_error": job.get("latest_context_error") if context is None else None,
                    "events": [],
                    "session_id": session_id,
                    "sequence": job.get("last_sequence"),
                }
                _json_line(Path(job["aligned_training_output"]), row)
                job["audio_windows"] = index + 1
                job["audio_frames"] = int(job["audio_frames"]) + WINDOW_FRAMES
                job["observed_windows"] = int(job["observed_windows"]) + 1
            job["status"] = "receiving"
            job["metadata_status"] = "running"
            self._write_status(job)

    def on_stop(self, payload: dict[str, Any]) -> str:
        session_id = str(payload["sessionId"])
        with self._lock:
            job = self._sessions.get(session_id)
            if job is not None:
                job["status"] = "stopped"
                job["metadata_status"] = "stopped"
                job["stopped_at"] = time.time()
                self._write_status(job)
        return "stopped"

    def status(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._sessions.get(session_id)
            if job is None:
                raise FileNotFoundError("No existe la sesión de entrenamiento del socket")
            return {key: value for key, value in job.items() if key != "pending_audio"}

    def active(self) -> dict[str, Any] | None:
        with self._lock:
            for job in reversed(list(self._sessions.values())):
                if job.get("status") not in {"stopped", "completed"}:
                    return {key: value for key, value in job.items() if key != "pending_audio"}
        return None

    @staticmethod
    def _redact_url(url: str) -> str:
        from urllib.parse import urlsplit, urlunsplit
        parsed = urlsplit(url)
        host = parsed.hostname or "<invalid-host>"
        if parsed.port:
            host += f":{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, parsed.query, ""))

    @staticmethod
    def _write_status(job: dict[str, Any]) -> None:
        path = Path(job["metadata_output"])
        safe = {key: value for key, value in job.items() if key != "pending_audio"}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            # A unique temporary file avoids collisions with a reader or with
            # another status write on Windows. The target is replaced only
            # after the complete JSON document has been closed.
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f"{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                handle.write(json.dumps(safe, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            os.replace(temporary_name, path)
            temporary_name = None
        finally:
            if temporary_name:
                try:
                    Path(temporary_name).unlink(missing_ok=True)
                except OSError:
                    pass


__all__ = ["SocketTrainingBridge"]
