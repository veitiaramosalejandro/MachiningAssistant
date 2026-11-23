"""Local web monitor for MachiningMonitor recordings and VAE training.

The browser never receives the MachiningMonitor API key. The backend reads it
from MACHINING_MONITOR_API_KEY and only exposes read-only service data plus
explicitly requested recording imports and training jobs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import mimetypes
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlparse, urlsplit, urlunsplit
import uuid
import wave

from persistence import health as storage_health
from persistence import persist_import, persist_training
from persistence import training_history as storage_training_history
from Integration.rtsp_cnc_metadata import StreamCncMetadata, build_stream_feature_vector
from Integration.rtsp_training_dataset import build_rtsp_dataset


PROJECT_ROOT = Path(__file__).resolve().parent
WEB_SOURCE_ROOT = PROJECT_ROOT / "web_monitor"
WEB_ROOT = WEB_SOURCE_ROOT / "dist" if (WEB_SOURCE_ROOT / "dist" / "index.html").is_file() else WEB_SOURCE_ROOT
RUNS_ROOT = PROJECT_ROOT / "runs" / "monitor_training"
INFERENCE_ROOT = PROJECT_ROOT / "runs" / "inference"
PYTHON = Path(os.environ.get("MACHINING_PYTHON", str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")))


def load_project_env(path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE settings without printing or modifying secrets."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


PROJECT_ENV = load_project_env(PROJECT_ROOT / ".env")


def setting(name: str, default: str = "") -> str:
    """Environment variables override the project .env file."""
    return os.environ.get(name, PROJECT_ENV.get(name, default))


def env_path(name: str, default: Path) -> Path:
    return Path(setting(name, str(default))).expanduser().resolve()


SERVICE_URL = setting("MACHINING_MONITOR_URL", "http://localhost:8888").rstrip("/")
API_KEY = setting("MACHINING_MONITOR_API_KEY")
DATA_ROOT = env_path(
    "MACHINING_MONITOR_DATA_ROOT",
    PROJECT_ROOT.parent / "REP_MachiningMonitor" / "machiningmonitor-data",
)
FFMPEG = setting("MACHINING_FFMPEG", "ffmpeg")
FFPROBE = setting("MACHINING_FFPROBE", "ffprobe")
LOCAL_DATA_ROOT = PROJECT_ROOT / "Dataset" / "Data"
SAMPLE_RATE_HZ = 44100
WINDOW_DURATION_OPTIONS_MS = {50, 100, 200, 500}

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_rtsp_listener_jobs: dict[str, dict] = {}
_rtsp_listener_lock = threading.Lock()


def _safe_rtsp_url(value: object) -> tuple[str, str]:
    """Validate a local RTSP source and return (usable_url, redacted_url)."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("rtsp_url es obligatorio")
    raw = value.strip()
    parts = urlsplit(raw)
    if parts.scheme.lower() != "rtsp" or not parts.hostname:
        raise ValueError("rtsp_url debe ser una URL RTSP válida")
    if parts.hostname.lower().rstrip(".") not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("La fase de pruebas solo permite una fuente RTSP local")
    if parts.port is not None and not 1 <= parts.port <= 65535:
        raise ValueError("El puerto RTSP no es válido")
    redacted = urlunsplit((parts.scheme, parts.hostname + (f":{parts.port}" if parts.port else ""), parts.path, parts.query, ""))
    return raw, redacted


def _prefer_ipv4_loopback(url: str) -> str:
    """Avoid Windows localhost/IPv6 resolution for the local RTSP publisher."""
    parts = urlsplit(url)
    if (parts.hostname or "").lower() != "localhost":
        return url
    userinfo = ""
    if parts.username is not None:
        userinfo = quote(parts.username, safe="")
        if parts.password is not None:
            userinfo += ":" + quote(parts.password, safe="*")
        userinfo += "@"
    netloc = userinfo + "127.0.0.1"
    if parts.port:
        netloc += f":{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _rtsp_credentials(payload: dict) -> tuple[str, str]:
    """Resolve credentials without persisting or returning the password."""
    payload_username = str(payload.get("rtsp_username", "") or "").strip()
    payload_password = str(payload.get("rtsp_password", "") or "")
    # Prefer the project URL credentials over stale process-level variables.
    configured_url = PROJECT_ENV.get("MACHINING_RTSP_URL", "") or setting("MACHINING_RTSP_URL", "")
    configured_parts = urlsplit(configured_url)
    configured_username = unquote(configured_parts.username or "")
    configured_password = unquote(configured_parts.password or "")
    username = payload_username or configured_username or setting("MACHINING_RTSP_USERNAME", "")
    password = payload_password or configured_password or setting("MACHINING_RTSP_PASSWORD", "")
    return username.strip(), password


def validate_rtsp_training_source(payload: dict) -> dict:
    """Probe audio and metadata tracks without starting capture or training."""
    if str(payload.get("model_family", "CNN1D-VAE")) != "CNN1D-VAE":
        raise ValueError("Esta fase de pruebas está reservada para CNN1D-VAE")
    if int(payload.get("audio_window_seconds", 5)) != 5:
        raise ValueError("La ventana RTSP inicial debe ser de 5 segundos")
    if str(payload.get("label_source", "synthetic_audio_transform")) not in {"synthetic_audio_transform", "metadata_event", "manual_review"}:
        raise ValueError("label_source no está permitido")
    source, redacted = _safe_rtsp_url(payload.get("rtsp_url", "rtsp://127.0.0.1:8555/audio"))
    parts = urlsplit(source)
    username, password = _rtsp_credentials(payload)
    if username or password:
        if not username or not password:
            raise ValueError("rtsp_username y rtsp_password deben recibirse juntos")
        source = urlunsplit((parts.scheme, f"{quote(username, safe='')}:{quote(password, safe='*')}@{parts.hostname}{f':{parts.port}' if parts.port else ''}", parts.path, parts.query, ""))
    source = _prefer_ipv4_loopback(source)
    timeout_s = int(payload.get("probe_timeout_seconds", 15))
    if timeout_s not in {5, 10, 15, 30}:
        raise ValueError("probe_timeout_seconds no está entre las opciones permitidas")
    command = [
        FFPROBE, "-hide_banner", "-v", "error", "-rtsp_transport", "tcp",
        "-show_streams", "-show_format", "-of", "json", source,
    ]
    started = time.time()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe no está disponible; configure MACHINING_FFPROBE") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"La comprobación RTSP superó {timeout_s} segundos") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()[-500:]
        raise RuntimeError(f"ffprobe no pudo abrir la fuente RTSP: {(detail or 'error desconocido').replace(source, redacted)}")
    try:
        report = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe devolvió una respuesta no válida") from exc
    streams = report.get("streams", []) if isinstance(report, dict) else []
    audio = [item for item in streams if item.get("codec_type") == "audio"]
    metadata = [
        item for item in streams
        if item.get("codec_type") in {"data", "unknown"}
        or str(item.get("codec_name", "")).lower() == "x-machining-metadata"
        or "machining" in json.dumps(item, ensure_ascii=False).lower()
    ]
    public_streams = []
    for item in streams:
        public_streams.append({
            "index": item.get("index"), "codec_type": item.get("codec_type"),
            "codec_name": item.get("codec_name"), "sample_rate": item.get("sample_rate"),
            "channels": item.get("channels"), "channel_layout": item.get("channel_layout"),
            "tags": item.get("tags", {}),
        })
    return {
        "ok": bool(audio and metadata),
        "rtsp_url": redacted,
        "checked_in_seconds": round(time.time() - started, 3),
        "model_family": "CNN1D-VAE",
        "audio_track_detected": bool(audio),
        "metadata_track_detected": bool(metadata),
        "metadata_contract": "x-machining-metadata / MAM1 framed JSON",
        "audio_expectations": {"sample_rate_hz": 44100, "channels": 2, "pcm": "PCM16LE"},
        "streams": public_streams,
        "next_step": "La captura debe persistir audio, metadata y etiquetas alineadas antes de iniciar el entrenamiento." if audio and metadata else "La fuente no expone simultáneamente audio y metadata; no se puede preparar un dataset multimodal.",
    }


def _redact_rtsp_text(value: str, source: str, redacted: str) -> str:
    """Remove RTSP userinfo from text returned by the listener."""
    return str(value or "").replace(source, redacted)[-12000:]


def start_rtsp_listener(payload: dict) -> dict:
    """Start the existing audio RTSP receiver as a web-controlled session."""
    with _rtsp_listener_lock:
        for active in _rtsp_listener_jobs.values():
            audio_process = active.get("process")
            metadata_process = active.get("metadata_process")
            if ((audio_process is not None and audio_process.poll() is None)
                    or (metadata_process is not None and metadata_process.poll() is None)):
                raise RuntimeError(
                    f"Ya existe una escucha RTSP activa ({active.get('listener_id')}); deténgala antes de iniciar otra"
                )
    source, redacted = _safe_rtsp_url(payload.get("rtsp_url", "rtsp://127.0.0.1:8555/audio"))
    username, password = _rtsp_credentials(payload)
    if username or password:
        if not username or not password:
            raise ValueError("rtsp_username y rtsp_password deben recibirse juntos")
        parts = urlsplit(source)
        source = urlunsplit((parts.scheme, f"{quote(username, safe='')}:{quote(password, safe='*')}@{parts.hostname}{f':{parts.port}' if parts.port else ''}", parts.path, parts.query, ""))
    source = _prefer_ipv4_loopback(source)
    transport = str(payload.get("transport", "tcp"))
    if transport not in {"tcp", "udp"}:
        raise ValueError("transport no está permitido")
    training_policy = str(payload.get("training_policy", "continuous_capture_then_batch"))
    if training_policy not in {"continuous_capture_then_batch", "capture_only"}:
        raise ValueError("training_policy no está permitido")
    legacy_windows = payload.get("min_labeled_windows")
    min_training_seconds = int(payload.get("min_training_seconds", (int(legacy_windows) * 5 if legacy_windows is not None else 5)))
    if not 5 <= min_training_seconds <= 600:
        raise ValueError("min_training_seconds debe estar entre 5 segundos y 600 segundos")
    checkpoint = _latest_rtsp_checkpoint() or Path(setting("MACHINING_CHECKPOINT", "")).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("MACHINING_CHECKPOINT no está configurado o no existe")
    model = str(setting("MACHINING_MODEL", "cnn")).lower()
    if model != "cnn":
        raise ValueError("La escucha web de esta fase requiere el modelo CNN1D-VAE (MACHINING_MODEL=cnn)")
    checkpoint_contract = _checkpoint_contract(checkpoint)
    session_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    root = PROJECT_ROOT / "runs" / "rtsp" / "web_sessions" / session_id
    root.mkdir(parents=True, exist_ok=False)
    output = root / "live_predictions.jsonl"
    metadata_output = root / "session.metadata.json"
    metadata_status_output = root / "metadata.status.json"
    aligned_output = root / "aligned_training.jsonl"
    audio_windows_output = root / "audio_windows"
    log_path = root / "listener.log"
    metadata_log_path = root / "metadata.listener.log"
    env = os.environ.copy()
    env.update({
        "MACHINING_RTSP_URL": source,
        "MACHINING_RTSP_TRANSPORT": transport,
        "MACHINING_MODEL": model,
        "MACHINING_CHECKPOINT": str(checkpoint),
        "MACHINING_RTSP_OUTPUT": str(output),
        "MACHINING_RTSP_METADATA_OUTPUT": str(metadata_output),
        "MACHINING_RTSP_CONTEXT_STREAM": str(root / "stream_metadata.jsonl"),
        "MACHINING_RTSP_AUDIO_WINDOWS_OUTPUT": str(audio_windows_output),
        "MACHINING_METADATA_RTSP_URL": source,
        "MACHINING_METADATA_OUTPUT": str(root / "stream_metadata.jsonl"),
        "MACHINING_METADATA_STATUS_OUTPUT": str(metadata_status_output),
    })
    command = [str(PYTHON), str(PROJECT_ROOT / "Integration" / "rtsp_stream_service.py")]
    metadata_command = [str(PYTHON), str(PROJECT_ROOT / "Integration" / "rtsp_metadata_listener.py")]
    log_handle = log_path.open("w", encoding="utf-8")
    metadata_log_handle = metadata_log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
        metadata_process = subprocess.Popen(metadata_command, cwd=PROJECT_ROOT, env=env, stdout=metadata_log_handle, stderr=subprocess.STDOUT, text=True)
    except Exception:
        log_handle.close()
        metadata_log_handle.close()
        if "process" in locals() and process.poll() is None:
            process.terminate()
        raise
    log_handle.close()
    metadata_log_handle.close()
    job = {
        "listener_id": session_id, "status": "starting", "pid": process.pid,
        "rtsp_url": redacted, "transport": transport, "model_family": "CNN1D-VAE",
        "training_policy": training_policy, "min_training_seconds": min_training_seconds,
        "minimum_labeled_windows": (min_training_seconds + 4) // 5,
        "training_ready": False, "labeled_windows": 0,
        "checkpoint_contract": checkpoint_contract,
        "output": str(output), "metadata_output": str(metadata_output), "log": str(log_path),
        "metadata_stream_output": str(root / "stream_metadata.jsonl"),
        "metadata_status_output": str(metadata_status_output), "metadata_log": str(metadata_log_path),
        "aligned_training_output": str(aligned_output),
        "audio_windows_output": str(audio_windows_output),
        "training_request_output": str(root / "training_request.json"), "started_at": time.time(),
        "process": process, "metadata_process": metadata_process,
    }
    with _rtsp_listener_lock:
        _rtsp_listener_jobs[session_id] = job
    return _rtsp_listener_public(job)


def _checkpoint_contract(checkpoint: Path) -> dict:
    import torch
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return {
        "model_family": state.get("model_family"), "task": state.get("task"),
        "context_contract": state.get("context_contract", ""),
        "context_dim": int(state.get("context_dim", 0)),
        "context_features": list(state.get("context_normalizer", {}).get("feature_names", []))
        if isinstance(state.get("context_normalizer"), dict) else [],
        "class_names": list(state.get("class_names", [])),
    }


def _latest_rtsp_checkpoint() -> Path | None:
    """Prefer the newest completed multimodal RTSP model for live capture."""
    candidates = sorted(
        (PROJECT_ROOT / "runs" / "rtsp" / "web_sessions").glob("*/training_run*/best.pt"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        try:
            contract = _checkpoint_contract(candidate)
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        if (contract.get("model_family") == "CNN1D-VAE"
                and contract.get("task") == "classification"
                and contract.get("context_contract") == "rtsp_stream"
                and contract.get("context_dim") == 7
                and len(contract.get("class_names", [])) == 6):
            return candidate
    return None


def _read_json_lines(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _align_rtsp_training_windows(job: dict) -> dict:
    """Align observed events with audio windows without creating training labels.

    ``event.quality`` is a model/output annotation for the current workflow,
    not an input or a training target.  Keep the raw event available for
    diagnostics, but do not convert it into ``label`` or ``quality_score``.
    """
    predictions = _read_json_lines(Path(str(job.get("output", ""))))
    messages = _read_json_lines(Path(str(job.get("metadata_stream_output", ""))))
    session_metadata = _read_json_file(Path(str(job.get("metadata_output", ""))), {})
    occurred_at_utc = session_metadata.get("started_at_utc") if isinstance(session_metadata, dict) else None
    # The metadata endpoint replays several recordings when a client connects
    # and timestampSeconds restarts at zero for each recording.  Keep the
    # newest recording that has a complete CNC context; incomplete historical
    # records must not block a valid current training batch.
    timeline = []
    seen_recordings = set()
    for item in messages:
        if item.get("type") != "stream_start" or not item.get("recordingId"):
            continue
        recording_id = str(item["recordingId"])
        if recording_id in seen_recordings:
            continue
        seen_recordings.add(recording_id)
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        duration = item.get("durationSeconds")
        if duration is None:
            duration = metadata.get("DurationSeconds")
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            duration = None
        timeline.append({"recording_id": recording_id, "offset": 0.0, "duration": duration})
    all_snapshots = [item for item in messages if item.get("type") == "snapshot"]
    target = None
    for candidate in reversed(timeline):
        candidate_snapshots = [
            item for item in all_snapshots
            if item.get("recordingId") == candidate["recording_id"]
        ]
        for snapshot in candidate_snapshots:
            try:
                build_stream_feature_vector(StreamCncMetadata.from_message(snapshot).to_mapping())
                target = candidate
                break
            except (TypeError, ValueError):
                continue
        if target is not None:
            break
    if target is not None:
        active_recording_id = target["recording_id"]
        scoped_messages = [item for item in messages if item.get("recordingId") == active_recording_id]
    else:
        active_recording_id = None
        scoped_messages = messages
    snapshots = [item for item in scoped_messages if item.get("type") == "snapshot"]
    raw_events = [item for item in scoped_messages if item.get("type") == "event"]
    events = []
    seen_events = set()
    for item in raw_events:
        event = item.get("event") if isinstance(item.get("event"), dict) else {}
        event_id = event.get("id") or (
            item.get("recordingId"), item.get("timestampSeconds"),
            event.get("occurrence"), event.get("value"),
        )
        if event_id in seen_events:
            continue
        seen_events.add(event_id)
        events.append(item)
    aligned = []
    for prediction in predictions:
        try:
            start = float(prediction["start_s"])
            end = float(prediction["end_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if target is not None:
            local_start = start
            local_end = end
            duration = target.get("duration")
            if local_end <= 0 or (duration is not None and local_start >= float(duration)):
                continue
        else:
            local_start, local_end = start, end
        window_events = []
        for item in events:
            try:
                timestamp = float(item.get("timestampSeconds"))
            except (TypeError, ValueError):
                continue
            playback_timestamp = timestamp
            if start <= playback_timestamp < end:
                event = item.get("event") if isinstance(item.get("event"), dict) else {}
                window_events.append({
                    "timestamp_seconds": playback_timestamp,
                    "recording_id": item.get("recordingId"),
                    "occurrence": event.get("occurrence"),
                    "event": event,
                })
        if not window_events:
            continue
        previous = [
            item for item in snapshots
            if isinstance(item.get("timestampSeconds"), (int, float)) and item["timestampSeconds"] <= local_start
        ]
        raw_snapshot = previous[-1] if previous else None
        context = None
        context_error = None
        if raw_snapshot is not None:
            try:
                context = StreamCncMetadata.from_message(raw_snapshot).to_mapping()
            except (TypeError, ValueError) as exc:
                context_error = str(exc)
        aligned.append({
            "window_start_seconds": start, "window_end_seconds": end,
            "audio_prediction": {
                "predicted_label": prediction.get("predicted_label"),
                "confidence_uncalibrated": prediction.get("confidence_uncalibrated"),
            },
            "training_label": None,
            "training_label_source": None,
            "audio_wav_file": prediction.get("audio_window_file"),
            "events": window_events, "context": context,
            "context_error": context_error,
            "recording_id": active_recording_id,
            "recording_playback_offset_seconds": target["offset"] if target is not None else 0.0,
        })
    aligned_path = Path(str(job.get("aligned_training_output", "")))
    if aligned_path:
        aligned_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = aligned_path.with_name(aligned_path.name + ".tmp")
        temporary.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in aligned), encoding="utf-8")
        temporary.replace(aligned_path)
    # No training window is considered labeled until an external ground-truth
    # source is implemented.  In particular, event.quality is never used here.
    job["observed_windows"] = len(aligned)
    job["labeled_windows"] = 0
    job["labeled_duration_seconds"] = 0
    job["training_ready"] = False
    job["metadata_messages"] = len(messages)
    job["metadata_events"] = len(events)
    job["quality_labels"] = []
    job["quality_label_scores"] = {}
    job["metadata_snapshots"] = len(snapshots)
    return {"observed_windows": job["observed_windows"], "labeled_windows": 0, "labeled_duration_seconds": 0, "training_ready": False, "metadata_messages": len(messages), "metadata_events": len(events), "metadata_snapshots": len(snapshots), "quality_labels": []}


def _rtsp_stream_observation(job: dict) -> dict:
    """Return bounded, UI-safe observations from the active RTSP session."""
    messages = _read_json_lines(Path(str(job.get("metadata_stream_output", ""))))
    predictions = _read_json_lines(Path(str(job.get("output", ""))))
    type_counts = {}
    for item in messages:
        message_type = str(item.get("type", "unknown"))
        type_counts[message_type] = type_counts.get(message_type, 0) + 1
    def event_row(item: dict) -> dict:
        event = item.get("event") if isinstance(item.get("event"), dict) else {}
        parameters = event.get("parameters") if isinstance(event.get("parameters"), dict) else {}
        spindle = parameters.get("SpindleSpeed") if isinstance(parameters.get("SpindleSpeed"), dict) else {}
        feed = parameters.get("FeedRate") if isinstance(parameters.get("FeedRate"), dict) else {}
        return {
            "timestamp_seconds": item.get("timestampSeconds"),
            "recording_id": item.get("recordingId"),
            "sequence": item.get("sequence"),
            "occurrence": event.get("occurrence"),
            "value": event.get("value"),
            "machine_state": event.get("machine_state_label") or event.get("machine_state"),
            "machine_activity": event.get("machine_activity_label") or event.get("machine_activity"),
            "quality": event.get("quality"),
            "observations": event.get("observations"),
            "programmed_rpm": spindle.get("ProgrammedRpm"),
            "adjusted_rpm": spindle.get("AdjustedRpm"),
            "programmed_feed": feed.get("ProgrammedMmPerMin"),
            "adjusted_feed": feed.get("AdjustedMmPerMin"),
        }

    seen_count = int(job.get("metadata_observation_count", 0))
    grace_until = float(job.setdefault(
        "metadata_history_grace_until",
        time.time() + float(job.get("metadata_history_grace_seconds", 10)),
    ))
    history_initialized = bool(job.get("metadata_history_initialized", False))
    phase = str(job.get("metadata_phase", "historical_bootstrap"))
    history_recording_ids = set(job.get("history_recording_ids", []))
    historical_events = list(job.get("historical_events", []))
    current_events = list(job.get("current_events", []))
    for item in messages[seen_count:]:
        recording_id = str(item.get("recordingId") or "")
        # The publisher may reuse recordingId across a resumed/current stream.
        # Arrival phase is therefore authoritative after the initial history
        # window; recordingId is retained only as diagnostic information.
        is_historical = (not history_initialized) or time.time() < grace_until
        if is_historical and recording_id:
            history_recording_ids.add(recording_id)
        if item.get("type") == "event":
            (historical_events if is_historical else current_events).append(event_row(item))
        if not is_historical:
            phase = "current"
    if time.time() < grace_until:
        phase = "historical_bootstrap"
    elif phase != "current":
        phase = "waiting_new_recording"
    job["metadata_observation_count"] = len(messages)
    job["metadata_history_initialized"] = True
    job["metadata_phase"] = phase
    job["history_recording_ids"] = sorted(history_recording_ids)
    job["historical_events"] = historical_events[-100:]
    job["current_events"] = current_events[-100:]
    snapshots = [item for item in messages if item.get("type") == "snapshot"]
    latest_snapshot = snapshots[-1] if snapshots else None
    return {
        "message_counts": type_counts,
        "metadata_messages": len(messages),
        "metadata_events": len(historical_events) + len(current_events),
        "metadata_snapshots": len(snapshots),
        "historical_events": historical_events[-50:],
        "current_events": current_events[-50:],
        "metadata_phase": phase,
        "latest_snapshot": latest_snapshot,
        "recent_audio_predictions": predictions[-25:],
    }


def _rtsp_listener_public(job: dict) -> dict:
    process = job.get("process")
    metadata_process = job.get("metadata_process")
    if process is not None:
        return_code = process.poll()
        if return_code is not None and job.get("status") not in {"stopped", "failed"}:
            job["status"] = "completed" if return_code == 0 else "failed"
            job["return_code"] = return_code
        if return_code is not None and return_code != 0 and metadata_process is not None and metadata_process.poll() is None:
            # Audio is the primary listener. Do not leave a metadata child
            # running after FFprobe/FFmpeg has failed, otherwise the UI reports
            # a misleading mixed state: audio failed, metadata running.
            metadata_process.terminate()
            job["metadata_status"] = "stopped_audio_failed"
    if metadata_process is not None:
        metadata_return_code = metadata_process.poll()
        if job.get("metadata_status") != "stopped_audio_failed":
            job["metadata_status"] = "running" if metadata_return_code is None else ("completed" if metadata_return_code == 0 else "failed")
        job["metadata_return_code"] = metadata_return_code
    training_process = job.get("training_process")
    if training_process is not None:
        training_return_code = training_process.poll()
        if training_return_code is None:
            job["training_status"] = "running"
        else:
            job["training_status"] = "completed" if training_return_code == 0 else "failed"
            job["training_return_code"] = training_return_code
            training_output = Path(str(job.get("training_output", "")))
            job["training_checkpoint"] = str(training_output / "best.pt") if (training_output / "best.pt").is_file() else None
            request_path = Path(str(job.get("training_request_output", "")))
            if request_path.is_file():
                request = _read_json_file(request_path, {})
                if isinstance(request, dict) and request.get("status") == "running":
                    request["status"] = job["training_status"]
                    request["accepted"] = training_return_code == 0
                    request["training_return_code"] = training_return_code
                    request["training_checkpoint"] = job["training_checkpoint"]
                    request_path.write_bytes(json_bytes(request))
    _align_rtsp_training_windows(job)
    result = {key: value for key, value in job.items() if key not in {"process", "metadata_process", "training_process"}}
    metadata_path = Path(str(job.get("metadata_output", "")))
    result["session_metadata"] = _read_json_file(metadata_path, {}) if metadata_path.is_file() else None
    result["metadata_stream_status"] = _read_json_file(Path(str(job.get("metadata_status_output", ""))), {})
    result["stream_observation"] = _rtsp_stream_observation(job)
    result["training_note"] = (
        "El entrenamiento RTSP sintético fue solicitado; las etiquetas proceden de transformaciones controladas del audio."
        if job.get("training_status") in {"starting", "running", "completed"} else
        "El entrenamiento RTSP usa seis etiquetas sintéticas; event.quality no se usa como entrada ni como target."
    )
    log_path = Path(str(job.get("log", "")))
    result["log_tail"] = log_path.read_text(encoding="utf-8", errors="replace")[-12000:] if log_path.is_file() else ""
    training_log = Path(str(job.get("training_log", "")))
    result["training_log_tail"] = training_log.read_text(encoding="utf-8", errors="replace")[-12000:] if training_log.is_file() else ""
    dataset_report = Path(str(job.get("dataset_report", "")))
    result["dataset_report"] = _read_json_file(dataset_report, {}) if dataset_report.is_file() else None
    training_output = Path(str(job.get("training_output", "")))
    result["training_metrics"] = _read_json_file(training_output / "metrics.json", {}) if training_output.is_dir() else None
    result["training_checkpoint"] = job.get("training_checkpoint")
    return result


def rtsp_listener_status(listener_id: str) -> dict:
    if not listener_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in listener_id):
        raise ValueError("listener_id inválido")
    with _rtsp_listener_lock:
        job = _rtsp_listener_jobs.get(listener_id)
        if job is None:
            raise FileNotFoundError("No existe la sesión RTSP solicitada")
        return _rtsp_listener_public(job)


def active_rtsp_listener_status() -> dict:
    """Return the active listener so a refreshed browser can recover it."""
    with _rtsp_listener_lock:
        for listener_id, job in _rtsp_listener_jobs.items():
            process = job.get("process")
            metadata_process = job.get("metadata_process")
            if ((process is not None and process.poll() is None)
                    or (metadata_process is not None and metadata_process.poll() is None)):
                result = _rtsp_listener_public(job)
                result["active"] = True
                return result
    return {"active": False, "listener_id": None}


def stop_rtsp_listener(listener_id: str) -> dict:
    if not listener_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in listener_id):
        raise ValueError("listener_id inválido")
    with _rtsp_listener_lock:
        job = _rtsp_listener_jobs.get(listener_id)
        if job is None:
            raise FileNotFoundError("No existe la sesión RTSP solicitada")
        process = job.get("process")
        metadata_process = job.get("metadata_process")
        if process is not None and process.poll() is None:
            process.terminate()
            job["status"] = "stopped"
        if metadata_process is not None and metadata_process.poll() is None:
            metadata_process.terminate()
        return _rtsp_listener_public(job)


def _next_rtsp_training_paths(session_dir: Path) -> tuple[Path, Path, str]:
    """Return a non-destructive output folder for an RTSP training attempt."""
    first_output = session_dir / "training_run"
    first_log = session_dir / "training.log"
    if not first_output.exists() or not any(first_output.iterdir()):
        return first_output, first_log, ""
    for attempt in range(2, 1000):
        suffix = f"_attempt_{attempt:02d}"
        output = session_dir / f"training_run{suffix}"
        if not output.exists():
            return output, session_dir / f"training{suffix}.log", suffix
    raise RuntimeError("No hay un directorio disponible para otro intento de entrenamiento RTSP")


RTSP_EPOCHS_PER_BATCH = 60


def _previous_rtsp_checkpoint(session_dir: Path, attempt_suffix: str) -> Path | None:
    """Resolve the previous batch checkpoint for a continuation run."""
    if not attempt_suffix:
        return None
    attempt_number = int(attempt_suffix.rsplit("_", 1)[-1])
    previous_suffix = "" if attempt_number == 2 else f"_attempt_{attempt_number - 1:02d}"
    checkpoint = session_dir / f"training_run{previous_suffix}" / "last.pt"
    return checkpoint if checkpoint.is_file() else None


def _checkpoint_completed_epoch(checkpoint: Path) -> int:
    import torch

    value = torch.load(checkpoint, map_location="cpu", weights_only=True)
    training_state = value.get("training_state")
    if not isinstance(training_state, dict):
        raise ValueError(f"El checkpoint RTSP no permite continuar: {checkpoint}")
    return int(training_state.get("completed_epoch", 0))


def request_rtsp_training(payload: dict) -> dict:
    listener_id = str(payload.get("listener_id", ""))
    rtsp_listener_status(listener_id)
    with _rtsp_listener_lock:
        job = _rtsp_listener_jobs.get(listener_id)
        if job is None:
            raise FileNotFoundError("No existe la sesión RTSP solicitada")
    blockers = []
    if int(job.get("metadata_events", 0)) == 0 and int(job.get("metadata_snapshots", 0)) == 0:
        blockers.append("No se han recibido snapshots ni eventos x-machining-metadata")
    session_dir = Path(str(job.get("aligned_training_output", ""))).parent
    training_output, training_log, attempt_suffix = _next_rtsp_training_paths(session_dir)
    dataset_dir = session_dir / ("training_dataset" if not attempt_suffix else f"training_dataset{attempt_suffix}")
    resume_path = _previous_rtsp_checkpoint(session_dir, attempt_suffix)
    completed_epoch = _checkpoint_completed_epoch(resume_path) if resume_path is not None else 0
    target_epochs = completed_epoch + RTSP_EPOCHS_PER_BATCH
    try:
        dataset_report = build_rtsp_dataset(
            session_dir=session_dir,
            aligned_path=Path(str(job.get("aligned_training_output", ""))),
            output_dir=dataset_dir,
        )
    except Exception as exc:
        dataset_report = {"ready": False, "status": "error", "error": str(exc), "synthetic_only": True}
        job["training_error"] = str(exc)
        blockers.append(f"Falló la generación del lote sintético: {exc}")
    job["dataset_report"] = str(dataset_dir / "dataset_report.json")
    if not dataset_report.get("ready"):
        blockers.append("El dataset sintético necesita al menos tres ventanas RTSP válidas para train, val y test")
    contract = job.get("checkpoint_contract", {})
    report = {"accepted": False, "status": "blocked", "listener_id": listener_id,
              "requested_at": time.time(), "blockers": blockers,
              "labeled_windows": job.get("labeled_windows", 0),
              "labeled_duration_seconds": job.get("labeled_duration_seconds", 0),
              "aligned_training_output": job.get("aligned_training_output"),
              "training_request_output": job.get("training_request_output"),
              "checkpoint_contract": contract,
              "capture_checkpoint_is_only_for_capture": True,
              "dataset_report": dataset_report,
              "training_dataset": str(dataset_dir),
              "quality_labels": list(dataset_report.get("required_classes", [])),
              "quality_label_scores": {},
              "label_source": "synthetic_audio_transform",
              "synthetic_only": True,
              "event_quality_used_as_target": False}
    if not blockers:
        existing = job.get("training_process")
        if existing is not None and existing.poll() is None:
            report.update({"accepted": True, "status": "running", "training_id": job.get("training_id"),
                           "training_output": job.get("training_output"), "training_log": job.get("training_log")})
        else:
            device = str(payload.get("device", "auto"))
            if device not in {"auto", "cpu", "cuda"}:
                raise ValueError("device no está permitido")
            training_id = f"{listener_id}_rtsp_cnn{attempt_suffix}"
            command = [str(PYTHON), str(PROJECT_ROOT / "CNN1D-VAE" / "train_fourclass.py"),
                       "--data-dir", str(dataset_dir), "--csv", str(dataset_dir / "samples.csv"),
                       "--cnc-context", str(dataset_dir / "contexts.json"),
                       "--cnc-contract", "rtsp_stream",
                       "--output-dir", str(training_output), "--epochs", str(target_epochs),
                       "--device", device]
            if resume_path is not None:
                command.extend(["--resume", str(resume_path), "--allow-data-change"])
            log_handle = training_log.open("w", encoding="utf-8")
            try:
                training_process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=os.environ.copy(),
                                                    stdout=log_handle, stderr=subprocess.STDOUT, text=True)
            finally:
                log_handle.close()
            job.update({"training_id": training_id, "training_status": "running",
                        "training_process": training_process, "training_output": str(training_output),
                        "training_log": str(training_log)})
            report.update({"accepted": True, "status": "running", "training_id": training_id,
                           "training_output": str(training_output), "training_log": str(training_log),
                           "command": command, "epochs_per_batch": RTSP_EPOCHS_PER_BATCH,
                           "target_epoch": target_epochs, "resumed_from": str(resume_path) if resume_path else None})
    request_path = Path(str(job.get("training_request_output", "")))
    if request_path:
        request_path.write_bytes(json_bytes(report))
    return report


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def sequence_length_for_duration(duration_ms: int) -> tuple[int, float]:
    """Convert a UI duration to a CNN-compatible sample count."""
    if duration_ms not in WINDOW_DURATION_OPTIONS_MS:
        raise ValueError("window_duration_ms no está entre las opciones permitidas")
    requested = duration_ms / 1000 * SAMPLE_RATE_HZ
    sequence_length = max(64, int(round(requested / 64)) * 64)
    actual_ms = sequence_length / SAMPLE_RATE_HZ * 1000
    return sequence_length, actual_ms


def service_get(path: str) -> object:
    from urllib.request import Request, urlopen

    if not API_KEY:
        raise RuntimeError("MACHINING_MONITOR_API_KEY is not configured")
    request = Request(SERVICE_URL + path, headers={"X-Api-Key": API_KEY}, method="GET")
    with urlopen(request, timeout=20) as response:  # noqa: S310 - configured local service URL
        return json.loads(response.read().decode("utf-8"))


def safe_execution_id(value: str) -> str:
    if not value or len(value) > 160 or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in value):
        raise ValueError("execution_id has an invalid format")
    return value


def find_recording_dir(execution_id: str) -> Path | None:
    """Find a local artifact folder by metadata ExecutionId, without trusting API paths."""
    execution_id = safe_execution_id(execution_id)
    if not DATA_ROOT.is_dir():
        return None
    for metadata_path in DATA_ROOT.rglob("metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("ExecutionId") == execution_id:
            return metadata_path.parent
    return None


def wav_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as source:
        return {
            "sample_rate_hz": source.getframerate(),
            "channels": source.getnchannels(),
            "sample_width_bytes": source.getsampwidth(),
            "frames": source.getnframes(),
            "duration_s": source.getnframes() / source.getframerate(),
            "compression": source.getcomptype(),
            "size_bytes": path.stat().st_size,
        }


def local_recording_info(execution_id: str) -> dict:
    folder = find_recording_dir(execution_id)
    if folder is None:
        return {"available": False, "folder": None, "files": [], "audio": []}
    files = []
    audio = []
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        item = {"name": path.name, "size_bytes": path.stat().st_size}
        if path.suffix.lower() == ".wav":
            try:
                info = wav_info(path)
                item["audio"] = info
                audio.append({"name": path.name, **info})
            except (OSError, wave.Error):
                item["audio_error"] = "invalid_wav"
        files.append(item)
    diagnostics = None
    diagnostic_path = folder / "segment_diagnostics.json"
    if diagnostic_path.is_file():
        try:
            diagnostics = json.loads(diagnostic_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            diagnostics = {"error": "invalid_segment_diagnostics"}
    return {
        "available": True,
        "folder": str(folder),
        "files": files,
        "audio": audio,
        "diagnostics": diagnostics,
    }


def execution_summary(execution: dict) -> dict:
    result = dict(execution)
    result["local_recording"] = local_recording_info(str(execution.get("executionId", "")))
    result["training_candidate"] = (
        execution.get("state") == "completed"
        and result["local_recording"].get("available")
        and bool(result["local_recording"].get("audio"))
        and not (result["local_recording"].get("diagnostics") or {}).get("Issues")
        and "falha" not in str(execution.get("detail", "")).lower()
    )
    return result


def local_execution_summaries(machine_id: str) -> list[dict]:
    """Expose completed service artifacts even when API history has expired."""
    if not DATA_ROOT.is_dir():
        return []
    results = []
    for metadata_path in DATA_ROOT.rglob("metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        execution_id = metadata.get("ExecutionId")
        if not isinstance(execution_id, str) or not execution_id:
            continue
        if machine_id and metadata.get("MachineId") != machine_id:
            continue
        item = {
            "machineName": metadata.get("MachineName", ""),
            "machineId": metadata.get("MachineId", ""),
            "executionId": execution_id,
            "driverSessionId": metadata.get("DriverSessionId"),
            "captureStartedAtUtc": metadata.get("StartTime"),
            "captureEndedAtUtc": metadata.get("EndTime"),
            "state": metadata.get("RecordingState"),
            "detail": metadata.get("Detail"),
            "folder": str(metadata_path.parent),
            "durationSeconds": metadata.get("DurationSeconds"),
            "source": "local_storage",
        }
        results.append(execution_summary(item))
    return results


def write_split_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["recording_id", "source_recording_id", "split"])
        writer.writeheader()
        writer.writerows(rows)


def _local_source_id(metadata: dict, recording_id: str) -> str:
    """Return the acquisition identity used to prevent source leakage."""
    source = metadata.get("SourceRecording")
    if isinstance(source, dict) and isinstance(source.get("Folder"), str):
        value = source["Folder"].strip().replace("\\", "/").rstrip("/")
        if value:
            return value.split("/")[-1]
    return recording_id


def local_dataset_inventory() -> dict:
    """Inspect Dataset/Data without copying audio or changing project state."""
    recordings = []
    if LOCAL_DATA_ROOT.is_dir():
        for folder in sorted(LOCAL_DATA_ROOT.iterdir(), key=lambda item: item.name.lower()):
            if not folder.is_dir():
                continue
            metadata_path = folder / "metadata.json"
            wavs = sorted(folder.glob("audio_*.wav"))
            if not metadata_path.is_file() or not wavs:
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
                audio = wav_info(wavs[0])
            except (OSError, ValueError, json.JSONDecodeError, wave.Error):
                continue
            folder_id = "local-" + hashlib.sha1(folder.name.encode("utf-8")).hexdigest()[:16]
            recordings.append({
                "folder": folder.name,
                "recording_id": folder_id,
                "duration_s": audio["duration_s"],
                "sample_rate_hz": audio["sample_rate_hz"],
                "channels": audio["channels"],
                "source_group": _local_source_id(metadata, folder_id),
            })
    return {
        "root": str(LOCAL_DATA_ROOT),
        "recordings": recordings,
        "recording_count": len(recordings),
        "source_group_count": len({item["source_group"] for item in recordings}),
        "duration_s": sum(item["duration_s"] for item in recordings),
    }


def import_local_dataset(sample_rate: int = 44100) -> dict:
    """Prepare all valid Dataset/Data recordings as a reproducible training job."""
    if sample_rate != 44100:
        raise ValueError("La primera versión del entrenamiento requiere 44100 Hz")
    inventory = local_dataset_inventory()
    if inventory["recording_count"] < 3:
        raise ValueError("Dataset/Data requiere al menos tres grabaciones válidas")
    job_id = time.strftime("%Y%m%d_%H%M%S") + "_local_" + uuid.uuid4().hex[:8]
    root = RUNS_ROOT / job_id
    data_dir = root / "data"
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    imported = []
    try:
        source_groups = {}
        for item in inventory["recordings"]:
            source_groups.setdefault(item["source_group"], []).append(item)
        ordered_groups = sorted(source_groups)
        group_splits = {
            group: ("train", "val", "test")[index % 3]
            for index, group in enumerate(ordered_groups)
        }
        for item in inventory["recordings"]:
            source_dir = LOCAL_DATA_ROOT / item["folder"]
            recording_id = item["recording_id"]
            target_dir = data_dir / recording_id
            target_dir.mkdir(parents=True)
            source_wav = sorted(source_dir.glob("audio_*.wav"))[0]
            target_wav = target_dir / "audio_000000.wav"
            command = [
                FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(source_wav), "-ac", "2", "-ar", str(sample_rate),
                "-c:a", "pcm_s16le", str(target_wav),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
            if completed.returncode != 0 or not target_wav.is_file():
                raise RuntimeError(f"FFmpeg no pudo preparar {item['folder']}: {completed.stderr[-500:]}")
            source_metadata = source_dir / "metadata.json"
            shutil.copy2(source_metadata, target_dir / "metadata.json")
            for name in ("contexts.json", "events.json", "segment_diagnostics.json"):
                source = source_dir / name
                if source.is_file():
                    shutil.copy2(source, target_dir / name)
            source_id = item["source_group"]
            rows.append({
                "recording_id": recording_id,
                "source_recording_id": source_id,
                "split": group_splits[source_id],
            })
            imported.append({
                "source_folder": item["folder"],
                "recording_id": recording_id,
                "source_recording_id": source_id,
                "wav": str(target_wav),
                "wav_info": wav_info(target_wav),
            })
        split_csv = root / "dataset_splits.csv"
        write_split_csv(split_csv, rows)
        manifest = {
            "job_id": job_id,
            "source": "Dataset/Data",
            "source_root": str(LOCAL_DATA_ROOT),
            "sample_rate_hz": sample_rate,
            "data_dir": str(data_dir),
            "split_csv": str(split_csv),
            "recordings": imported,
            "source_groups": {
                split: sum(1 for value in group_splits.values() if value == split)
                for split in ("train", "val", "test")
            },
        }
        (root / "import_manifest.json").write_bytes(json_bytes(manifest))
        return manifest
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def import_one_local_recording(folder_name: str, sample_rate: int = 44100) -> dict:
    """Create a specialized temporal train/val/test dataset from one local WAV."""
    if sample_rate != 44100:
        raise ValueError("La primera versión del entrenamiento requiere 44100 Hz")
    item = next((value for value in local_dataset_inventory()["recordings"] if value["folder"] == folder_name), None)
    if item is None:
        raise FileNotFoundError("La grabación local seleccionada no existe o no contiene un WAV válido")
    source_dir = LOCAL_DATA_ROOT / folder_name
    source_wav = sorted(source_dir.glob("audio_*.wav"))[0]
    duration = float(item["duration_s"])
    if duration < 1.0:
        raise ValueError("La grabación es demasiado corta para crear tres segmentos temporales")
    boundaries = {
        "train": (0.0, duration * 0.70),
        "val": (duration * 0.70, duration * 0.85),
        "test": (duration * 0.85, duration),
    }
    job_id = time.strftime("%Y%m%d_%H%M%S") + "_single_" + uuid.uuid4().hex[:8]
    root = RUNS_ROOT / job_id
    data_dir = root / "data"
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    imported = []
    try:
        source_metadata = _read_json_file(source_dir / "metadata.json", {})
        base_id = item["recording_id"]
        for split, (start_s, end_s) in boundaries.items():
            recording_id = f"{base_id}-{split}"
            target_dir = data_dir / recording_id
            target_dir.mkdir(parents=True)
            target_wav = target_dir / "audio_000000.wav"
            command = [
                FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(source_wav), "-ss", f"{start_s:.6f}", "-t", f"{end_s - start_s:.6f}",
                "-ac", "2", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(target_wav),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
            if completed.returncode != 0 or not target_wav.is_file():
                raise RuntimeError(f"FFmpeg no pudo preparar el segmento {split}: {completed.stderr[-500:]}")
            metadata = dict(source_metadata)
            metadata.pop("SourceRecording", None)
            metadata["ExecutionId"] = recording_id
            metadata["SpecializedSourceRecording"] = {
                "folder": folder_name,
                "recording_id": base_id,
                "split": split,
                "start_seconds": start_s,
                "end_seconds": end_s,
                "split_policy": "70/15/15 temporal; no es una validacion independiente por grabacion",
            }
            (target_dir / "metadata.json").write_bytes(json_bytes(metadata))
            rows.append({"recording_id": recording_id, "source_recording_id": recording_id, "split": split})
            imported.append({
                "source_folder": folder_name, "recording_id": recording_id, "split": split,
                "start_seconds": start_s, "end_seconds": end_s,
                "wav": str(target_wav), "wav_info": wav_info(target_wav),
            })
        split_csv = root / "dataset_splits.csv"
        write_split_csv(split_csv, rows)
        manifest = {
            "job_id": job_id,
            "source": "Dataset/Data/single_recording",
            "source_root": str(LOCAL_DATA_ROOT),
            "source_folder": folder_name,
            "source_recording_id": base_id,
            "temporal_split": {"train": 0.70, "val": 0.15, "test": 0.15},
            "validation_note": "Train, validation y test son segmentos de una misma grabacion; no representan generalizacion entre sesiones.",
            "sample_rate_hz": sample_rate,
            "data_dir": str(data_dir), "split_csv": str(split_csv), "recordings": imported,
        }
        (root / "import_manifest.json").write_bytes(json_bytes(manifest))
        return manifest
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def import_recordings(execution_ids: list[str], sample_rate: int = 44100) -> dict:
    if len(execution_ids) < 3:
        raise ValueError("Seleccione al menos tres grabaciones completas para train, val y test")
    if sample_rate != 44100:
        raise ValueError("La primera versión del entrenamiento requiere 44100 Hz")
    job_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    root = RUNS_ROOT / job_id
    data_dir = root / "data"
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    imported = []
    try:
        for index, execution_id in enumerate(execution_ids):
            execution_id = safe_execution_id(execution_id)
            source_dir = find_recording_dir(execution_id)
            if source_dir is None:
                raise FileNotFoundError(f"No existe almacenamiento local para {execution_id}")
            wavs = sorted(source_dir.glob("audio_*.wav"))
            if not wavs:
                raise FileNotFoundError(f"La ejecución {execution_id} no contiene audio WAV")
            # Keep the dataset recording ID equal to the source execution ID.
            # This agrees with company_data.py when the copied metadata has no
            # SourceRecording wrapper and preserves a direct trace to Monitor.
            recording_id = execution_id
            target_dir = data_dir / recording_id
            target_dir.mkdir(parents=True)
            target_wav = target_dir / "audio_000000.wav"
            source_wav = wavs[0]
            command = [
                FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(source_wav), "-ac", "2", "-ar", str(sample_rate),
                "-c:a", "pcm_s16le", str(target_wav),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
            if completed.returncode != 0 or not target_wav.is_file():
                raise RuntimeError(f"FFmpeg no pudo preparar {execution_id}: {completed.stderr[-500:]}")
            metadata_path = source_dir / "metadata.json"
            if metadata_path.is_file():
                shutil.copy2(metadata_path, target_dir / "metadata.json")
            else:
                (target_dir / "metadata.json").write_text(
                    json.dumps({"ExecutionId": execution_id}, indent=2) + "\n", encoding="utf-8"
                )
            for name in ("contexts.json", "events.json", "driver_messages.json", "segment_diagnostics.json"):
                source = source_dir / name
                if source.is_file():
                    shutil.copy2(source, target_dir / name)
            split = ("train", "val", "test")[index % 3]
            # Each temporal segment is a distinct dataset recording so the
            # split validator does not treat one session as cross-split leakage.
            # The original execution remains in the manifest and metadata for traceability.
            rows.append({"recording_id": recording_id, "source_recording_id": recording_id, "split": split})
            imported.append({"execution_id": execution_id, "recording_id": recording_id, "wav": str(target_wav), "wav_info": wav_info(target_wav)})
        split_csv = root / "dataset_splits.csv"
        write_split_csv(split_csv, rows)
        manifest = {"job_id": job_id, "sample_rate_hz": sample_rate, "data_dir": str(data_dir), "split_csv": str(split_csv), "recordings": imported}
        (root / "import_manifest.json").write_bytes(json_bytes(manifest))
        return manifest
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def _slice_timed_document(document: dict, collection: str, start_s: float, end_s: float, carry_forward: bool = False) -> dict:
    """Shift context/event timestamps into a temporal recording segment."""
    if not isinstance(document, dict) or not isinstance(document.get(collection), list):
        return document if isinstance(document, dict) else {}
    entries = [item for item in document[collection] if isinstance(item, dict)]
    selected = []
    previous = None
    for item in entries:
        timestamp = float(item.get("timestamp_seconds", 0) or 0)
        if timestamp <= start_s:
            previous = item
        if start_s <= timestamp <= end_s:
            selected.append(item)
    if carry_forward and previous is not None and previous not in selected:
        selected.insert(0, previous)
    result = dict(document)
    result[collection] = []
    for item in selected:
        copied = dict(item)
        copied["timestamp_seconds"] = max(0.0, float(item.get("timestamp_seconds", 0) or 0) - start_s)
        result[collection].append(copied)
    return result


def import_one_service_recording(execution_id: str, sample_rate: int = 44100) -> dict:
    """Prepare one service recording as a temporal train/val/test dataset."""
    if sample_rate != 44100:
        raise ValueError("La primera versión del entrenamiento requiere 44100 Hz")
    execution_id = safe_execution_id(execution_id)
    source_dir = find_recording_dir(execution_id)
    if source_dir is None:
        raise FileNotFoundError(f"No existe almacenamiento local para {execution_id}")
    wavs = sorted(source_dir.glob("audio_*.wav"))
    if not wavs:
        raise FileNotFoundError(f"La ejecución {execution_id} no contiene audio WAV")
    duration = float(wav_info(wavs[0])["duration_s"])
    if duration < 1.0:
        raise ValueError("La grabación es demasiado corta para crear tres segmentos temporales")
    boundaries = {
        "train": (0.0, duration * 0.70),
        "val": (duration * 0.70, duration * 0.85),
        "test": (duration * 0.85, duration),
    }
    job_id = time.strftime("%Y%m%d_%H%M%S") + "_single_service_" + uuid.uuid4().hex[:8]
    root = RUNS_ROOT / job_id
    data_dir = root / "data"
    root.mkdir(parents=True, exist_ok=False)
    imported, rows = [], []
    try:
        source_metadata = _read_json_file(source_dir / "metadata.json", {"ExecutionId": execution_id})
        contexts = _read_json_file(source_dir / "contexts.json", {})
        events = _read_json_file(source_dir / "events.json", {})
        playback = _read_json_file(source_dir / "playback.json", {})
        for split, (start_s, end_s) in boundaries.items():
            recording_id = f"{execution_id}-{split}"
            target_dir = data_dir / recording_id
            target_dir.mkdir(parents=True)
            target_wav = target_dir / "audio_000000.wav"
            command = [
                FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(wavs[0]),
                "-ss", f"{start_s:.6f}", "-t", f"{end_s - start_s:.6f}",
                "-ac", "2", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(target_wav),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
            if completed.returncode != 0 or not target_wav.is_file():
                raise RuntimeError(f"FFmpeg no pudo preparar el segmento {split}: {completed.stderr[-500:]}")
            metadata = dict(source_metadata)
            metadata.pop("SourceRecording", None)
            metadata["ExecutionId"] = recording_id
            metadata["SpecializedSourceRecording"] = {
                "source": "MachiningMonitor",
                "execution_id": execution_id,
                "source_folder": str(source_dir),
                "split": split,
                "start_seconds": start_s,
                "end_seconds": end_s,
                "split_policy": "70/15/15 temporal; misma sesión de servicio",
            }
            (target_dir / "metadata.json").write_bytes(json_bytes(metadata))
            (target_dir / "contexts.json").write_bytes(json_bytes(_slice_timed_document(contexts, "contexts", start_s, end_s, True)))
            (target_dir / "events.json").write_bytes(json_bytes(_slice_timed_document(events, "annotations", start_s, end_s, False)))
            (target_dir / "playback.json").write_bytes(json_bytes(playback))
            # Temporal segments are distinct dataset recordings for split validation.
            # The original execution_id remains in metadata and the manifest.
            rows.append({"recording_id": recording_id, "source_recording_id": recording_id, "split": split})
            imported.append({
                "execution_id": execution_id, "source_folder": source_dir.name, "recording_id": recording_id,
                "split": split, "start_seconds": start_s, "end_seconds": end_s,
                "wav": str(target_wav), "wav_info": wav_info(target_wav),
            })
        split_csv = root / "dataset_splits.csv"
        write_split_csv(split_csv, rows)
        manifest = {
            "job_id": job_id, "source": "MachiningMonitor/single_recording", "source_root": str(source_dir.parent),
            "source_folder": source_dir.name, "source_recording_id": execution_id,
            "temporal_split": {"train": 0.70, "val": 0.15, "test": 0.15},
            "validation_note": "Los tres splits proceden de una misma sesión del servicio; no representan generalización entre sesiones.",
            "sample_rate_hz": sample_rate, "data_dir": str(data_dir), "split_csv": str(split_csv), "recordings": imported,
        }
        (root / "import_manifest.json").write_bytes(json_bytes(manifest))
        return manifest
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def _persist_import_best_effort(manifest: dict) -> dict:
    try:
        manifest["persistence"] = persist_import(manifest)
    except Exception as exc:
        manifest["persistence"] = {"ok": False, "error": str(exc)}
    return manifest


def train_worker(job_id: str, args: list[str], output_dir: Path) -> None:
    # The trainer requires its output directory to be empty at startup. Keep
    # the monitor log beside that directory, never inside it.
    log_path = output_dir.parent / "training.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(args, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        with _jobs_lock:
            _jobs[job_id]["pid"] = process.pid
            _jobs[job_id]["status"] = "running"
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            with _jobs_lock:
                _jobs[job_id]["log_tail"] = (_jobs[job_id].get("log_tail", "") + line)[-12000:]
        return_code = process.wait()
    with _jobs_lock:
        _jobs[job_id].update({"status": "completed" if return_code == 0 else "failed", "return_code": return_code, "finished_at": time.time()})
    if return_code == 0:
        _persist_completed_training(job_id, output_dir)


def _read_json_file(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return default


def _float_or_none(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _context_at(context_doc: dict, metadata: dict, timestamp_s: float) -> dict:
    entries = context_doc.get("contexts", []) if isinstance(context_doc, dict) else []
    candidates = [item for item in entries if isinstance(item, dict) and float(item.get("timestamp_seconds", 0) or 0) <= timestamp_s]
    selected = max(candidates, key=lambda item: float(item.get("timestamp_seconds", 0) or 0)) if candidates else None
    context = selected.get("context", {}) if selected else metadata.get("Context", {})
    if not isinstance(context, dict):
        context = {}
    tool = context.get("Tool", {})
    parameters = context.get("Parameters", {})
    if not isinstance(tool, dict):
        tool = {"name": str(tool)} if tool else {}
    if not isinstance(parameters, dict):
        parameters = {}
    spindle = parameters.get("SpindleSpeed", {})
    feed = parameters.get("FeedRate", {})
    if not isinstance(spindle, dict):
        spindle = {}
    if not isinstance(feed, dict):
        feed = {}
    return {
        "operation": context.get("Operation"),
        "work": context.get("Work"),
        "part": context.get("Part"),
        "material": context.get("PartMaterial"),
        "tool": tool.get("Tool") or tool.get("name"),
        "tool_type": tool.get("Type"),
        "rpm": spindle.get("AdjustedRpm") or spindle.get("ProgrammedRpm"),
        "feed_rate": feed.get("AdjustedMmPerMin") or feed.get("ProgrammedMmPerMin"),
        "context_timestamp_s": selected.get("timestamp_seconds") if selected else None,
    }


def _learning_statistics(report: dict) -> dict:
    """Return a comparable 0-100 learning indicator for one VAE run."""
    def number(value):
        try:
            value = float(value)
            return value if math.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    def nested(section: str, key: str):
        value = report.get(section, {})
        return number(value.get(key)) if isinstance(value, dict) else None

    history = report.get("history", [])
    best_epoch = int(report.get("best_epoch") or 0)
    evaluated = [
        item for item in history
        if isinstance(item, dict) and item.get("validation_evaluated")
        and number(item.get("val_reconstruction_loss")) is not None
    ]
    best = next((item for item in evaluated if int(item.get("epoch") or 0) == best_epoch), None)
    if best is None and evaluated:
        best = min(evaluated, key=lambda item: float(item["val_reconstruction_loss"]))
        best_epoch = int(best.get("epoch") or 0)
    first_train = next((number(item.get("train_reconstruction_loss")) for item in history if isinstance(item, dict) and number(item.get("train_reconstruction_loss")) is not None), None)
    before = report.get("before", {})
    baseline_train = (number(before.get("train", {}).get("reconstruction_loss"))
                      if isinstance(before, dict) and isinstance(before.get("train"), dict) else None) or first_train
    baseline_val = (number(before.get("val", {}).get("reconstruction_loss"))
                    if isinstance(before, dict) and isinstance(before.get("val"), dict) else None)
    current_train = number(best.get("train_reconstruction_loss")) if best else (
        number(report.get("after", {}).get("train", {}).get("reconstruction_loss"))
        if isinstance(report.get("after"), dict) and isinstance(report.get("after", {}).get("train"), dict) else None
    )
    current_val = number(best.get("val_reconstruction_loss")) if best else (
        number(report.get("after", {}).get("val", {}).get("reconstruction_loss"))
        if isinstance(report.get("after"), dict) and isinstance(report.get("after", {}).get("val"), dict) else None
    )

    def improvement(baseline, current):
        if baseline is None or current is None or baseline <= 0:
            return None
        return max(0.0, min(100.0, (1.0 - current / baseline) * 100.0))

    train_percent = improvement(baseline_train, current_train)
    validation_percent = improvement(baseline_val, current_val)
    components = [value for value in (train_percent, validation_percent) if value is not None]
    if train_percent is not None and validation_percent is not None:
        global_percent = train_percent * 0.4 + validation_percent * 0.6
    else:
        global_percent = components[0] if components else 0.0
    gap = None
    if current_train is not None and current_val is not None and current_train > 0:
        gap = max(0.0, (current_val - current_train) / current_train * 100.0)
    epochs = int(report.get("epochs") or (history[-1].get("epoch", 0) if history else 0))
    return {
        "learning_percent": round(global_percent, 2),
        "train_improvement_percent": round(train_percent, 2) if train_percent is not None else None,
        "validation_improvement_percent": round(validation_percent, 2) if validation_percent is not None else None,
        "epoch_progress_percent": round(max(0.0, min(100.0, best_epoch / epochs * 100.0)), 2) if epochs else 0.0,
        "generalization_gap_percent": round(gap, 2) if gap is not None else None,
        "best_epoch": best_epoch,
        "completed_epoch": epochs,
        "baseline_train_mse": baseline_train,
        "best_train_mse": current_train,
        "baseline_validation_mse": baseline_val,
        "best_validation_mse": current_val,
        "formula": "0.4 * train_improvement + 0.6 * validation_improvement; capped to 0-100",
        "interpretation": "Indicador relativo de reducción del error; no es accuracy ni porcentaje de detección.",
    }


def _training_result(output_dir: Path) -> dict:
    report = _read_json_file(output_dir / "test_report.json", {})
    config = _read_json_file(output_dir / "training_config.json", {})
    data_dir = Path(config.get("data_dir", ""))
    metadata_cache = {}
    context_cache = {}
    events_cache = {}
    playback_cache = {}

    def recording_context(recording_id: str) -> tuple[dict, dict, list, dict]:
        if recording_id not in metadata_cache:
            folder = data_dir / recording_id
            metadata_cache[recording_id] = _read_json_file(folder / "metadata.json", {})
            context_cache[recording_id] = _read_json_file(folder / "contexts.json", {})
            event_doc = _read_json_file(folder / "events.json", {})
            events_cache[recording_id] = event_doc.get("annotations", []) if isinstance(event_doc, dict) else []
            playback_cache[recording_id] = _read_json_file(folder / "playback.json", {})
        return metadata_cache[recording_id], context_cache[recording_id], events_cache[recording_id], playback_cache[recording_id]

    rows = []
    prediction_path = output_dir / "window_predictions.csv"
    if prediction_path.is_file():
        with prediction_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    start_s = float(row.get("start_s", 0))
                    end_s = float(row.get("end_s", start_s))
                    score = float(row.get("best_mse_standardized", row.get("trained_mse_standardized", "nan")))
                except (TypeError, ValueError):
                    continue
                recording_id = row.get("recording_id", "")
                metadata, context_doc, events, playback = recording_context(recording_id)
                playback_offset_ms = float(playback.get("AudioOffsetMs", 0) or 0) if isinstance(playback, dict) else 0
                context_time_s = (start_s + end_s) / 2 + playback_offset_ms / 1000
                overlap_events = []
                for event in events:
                    if not isinstance(event, dict):
                        continue
                    try:
                        event_s = float(event.get("timestamp_seconds", 0) or 0)
                    except (TypeError, ValueError):
                        continue
                    if start_s <= event_s <= end_s:
                        overlap_events.append({
                            "timestamp_seconds": event_s,
                            "quality": event.get("quality"),
                            "machine_state": event.get("machine_state_label") or event.get("machine_state"),
                            "observations": event.get("observations"),
                        })
                enriched = {
                    "recording_id": recording_id,
                    "source_recording_id": row.get("source_recording_id"),
                    "split": row.get("split"),
                    "start_s": start_s,
                    "end_s": end_s,
                    "score": score,
                    "baseline_state": row.get("baseline_state"),
                    "baseline_threshold": _float_or_none(row.get("baseline_threshold")),
                    "baseline_threshold_source": row.get("baseline_threshold_source"),
                    "operation_context": _context_at(context_doc, metadata, context_time_s),
                    "events": overlap_events,
                    "playback_offset_ms": playback_offset_ms,
                    "source_start_seconds": (
                        (metadata.get("SourceRecording") or {}).get("StartSeconds")
                        if isinstance(metadata.get("SourceRecording"), dict)
                        else ((metadata.get("SpecializedSourceRecording") or {}).get("start_seconds")
                              if isinstance(metadata.get("SpecializedSourceRecording"), dict) else None)
                    ),
                    "recording_start": metadata.get("StartTime"),
                }
                rows.append(enriched)
    finite_rows = [row for row in rows if math.isfinite(row["score"])]
    baseline_report = _read_json_file(output_dir / "baseline_report.json", report.get("baseline", {}))
    global_stats = baseline_report.get("global", {}) if isinstance(baseline_report, dict) else {}
    threshold = global_stats.get("p99")
    for row in finite_rows:
        row["above_train_p99"] = threshold is not None and row["score"] > float(threshold)
    top_windows = sorted(finite_rows, key=lambda row: row["score"], reverse=True)[:100]
    context_counts = {}
    for row in finite_rows:
        operation = row["operation_context"].get("operation") or "Sin contexto"
        bucket = context_counts.setdefault(operation, {"operation": operation, "windows": 0, "max_score": 0.0, "mean_score": 0.0})
        bucket["windows"] += 1
        bucket["max_score"] = max(bucket["max_score"], row["score"])
        bucket["mean_score"] += row["score"]
    for bucket in context_counts.values():
        bucket["mean_score"] /= bucket["windows"]
    return {
        "report": report,
        "learning_statistics": _learning_statistics(report),
        "history": report.get("history", []),
        "baseline": report.get("mean_reconstruction_baseline_mse", {}),
        "baseline_report": baseline_report,
        "best_epoch": report.get("best_epoch"),
        "best_checkpoint": report.get("best_checkpoint"),
        "score_reference_train_p99": threshold,
        "score_reference_note": "Baseline de train versionado; p95 indica revisión y p99 anomalía estadística. No es una probabilidad de fallo industrial.",
        "context_counts": sorted(context_counts.values(), key=lambda item: item["max_score"], reverse=True),
        "top_windows": top_windows,
        "window_count": len(finite_rows),
    }


def _persist_completed_training(training_id: str, output_dir: Path) -> None:
    """Persist a completed run without turning storage outages into train failures."""
    with _jobs_lock:
        job = dict(_jobs.get(training_id, {}))
    persistence_state: dict[str, object] = {}
    try:
        result = _training_result(output_dir)
        configuration = job.get("configuration", {})
        model = str(job.get("model", "cnn"))
        manifest = _read_json_file(output_dir.parent.parent / "import_manifest.json", {})
        data_dir = Path(manifest.get("data_dir", ""))
        from vector_index import index_training

        windows, vector_state = index_training(training_id, output_dir, data_dir, model, configuration)
        persistence_state["qdrant"] = vector_state
        persistence_state["postgres"] = persist_training(training_id, job, result, windows)["ok"]
    except Exception as exc:
        persistence_state["ok"] = False
        persistence_state["error"] = str(exc)
    with _jobs_lock:
        _jobs.setdefault(training_id, {})["persistence"] = persistence_state


def start_training(payload: dict) -> dict:
    job_id = str(payload.get("import_job_id", ""))
    if not job_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in job_id):
        raise ValueError("import_job_id inválido")
    model = str(payload.get("model", "cnn")).lower()
    if model not in {"cnn", "lstm"}:
        raise ValueError("model debe ser cnn o lstm")
    allowed = {
        "epochs": {20, 40, 60, 100, 120, 200, 350},
        "window_duration_ms": WINDOW_DURATION_OPTIONS_MS,
        "latent_dim": {8, 16, 32},
        "batch_size": {8, 16, 32},
        "learning_rate": {0.0001, 0.0003, 0.001},
        "beta": {0.0001, 0.001, 0.01},
        "eval_every": {5, 10, 20},
        "early_stopping_patience": {0, 3, 5, 10, 15},
        "device": {"auto", "cuda", "cpu"},
    }
    def selected(name: str, default, converter):
        value = converter(payload.get(name, default))
        if value not in allowed[name]:
            raise ValueError(f"{name} no está entre las opciones permitidas")
        return value

    epochs = selected("epochs", 20, int)
    requested_duration_ms = selected("window_duration_ms", 50, int)
    sequence_length, actual_duration_ms = sequence_length_for_duration(requested_duration_ms)
    latent_dim = selected("latent_dim", 16, int)
    batch_size = selected("batch_size", 8, int)
    learning_rate = selected("learning_rate", 0.0003, float)
    beta = selected("beta", 0.001, float)
    eval_every = selected("eval_every", 10, int)
    early_stopping_patience = selected("early_stopping_patience", 5, int)
    device = selected("device", "auto", str)
    root = RUNS_ROOT / job_id
    manifest_path = root / "import_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("No existe la importación seleccionada")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = root / "training" / model
    output_dir.mkdir(parents=True, exist_ok=False)
    script = PROJECT_ROOT / ("CNN1D-VAE" if model == "cnn" else "LSTM-VAE") / "run_demo.py"
    args = [
        str(PYTHON), str(script), "--data-dir", manifest["data_dir"], "--split-csv", manifest["split_csv"],
        "--output-dir", str(output_dir), "--epochs", str(epochs), "--all-windows",
        "--sequence-length", str(sequence_length), "--latent-dim", str(latent_dim),
        "--batch-size", str(batch_size), "--learning-rate", str(learning_rate), "--beta", str(beta),
        "--logvar-min", "-8", "--logvar-max", "8",
        "--eval-every", str(eval_every), "--device", device,
        "--early-stopping-patience", str(early_stopping_patience),
    ]
    training_id = f"{job_id}_{model}"
    job = {"training_id": training_id, "import_job_id": job_id, "model": model, "status": "starting", "args": args[1:], "configuration": {"all_windows": True, "window_duration_ms": requested_duration_ms, "actual_window_duration_ms": round(actual_duration_ms, 3), "sequence_length": sequence_length, "latent_dim": latent_dim, "batch_size": batch_size, "learning_rate": learning_rate, "beta": beta, "logvar_min": -8.0, "logvar_max": 8.0, "eval_every": eval_every, "early_stopping_patience": early_stopping_patience, "device": device}, "started_at": time.time(), "output_dir": str(output_dir), "log_tail": ""}
    with _jobs_lock:
        _jobs[training_id] = job
    threading.Thread(target=train_worker, args=(training_id, args, output_dir), daemon=True).start()
    return job


def _safe_workspace_path(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    allowed_roots = [DATA_ROOT, LOCAL_DATA_ROOT, RUNS_ROOT, INFERENCE_ROOT]
    if not any(path == root or root in path.parents for root in allowed_roots):
        raise ValueError(f"{label} está fuera de las carpetas permitidas")
    return path


def _training_output(training_id: str) -> Path:
    if not training_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in training_id):
        raise ValueError("training_id inválido")
    with _jobs_lock:
        job = dict(_jobs.get(training_id, {}))
    if job.get("output_dir"):
        output = Path(job["output_dir"]).resolve()
    else:
        model = training_id.rsplit("_", 1)[-1]
        if model not in {"cnn", "lstm"}:
            raise ValueError("training_id no contiene un modelo válido")
        job_id = training_id[:-(len(model) + 1)]
        output = (RUNS_ROOT / job_id / "training" / model).resolve()
    if not output.is_dir() or not (output / "best.pt").is_file():
        raise FileNotFoundError("El entrenamiento no tiene un best.pt disponible")
    return output


def inference_models() -> dict:
    models = []
    if RUNS_ROOT.is_dir():
        for checkpoint in sorted(RUNS_ROOT.glob("*/training/*/best.pt"), key=lambda item: item.stat().st_mtime, reverse=True):
            model = checkpoint.parent.name
            job_id = checkpoint.parent.parent.parent.name
            models.append({
                "training_id": f"{job_id}_{model}", "model": model,
                "checkpoint": str(checkpoint),
                "report": _read_json_file(checkpoint.parent / "test_report.json", {}),
            })
    return {"models": models[:50]}


def _inference_worker(inference_id: str, payload: dict, output_dir: Path) -> None:
    try:
        from reconstruction_inference import analyze_recording

        output = _training_output(str(payload["training_id"]))
        checkpoint = output / "best.pt"
        wav_path = _safe_workspace_path(str(payload["wav_path"]), "WAV")
        recording_dir = _safe_workspace_path(str(payload.get("recording_dir") or wav_path.parent), "grabación")
        previous = sorted(INFERENCE_ROOT.glob("*/analysis_report.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:20]
        report = analyze_recording(
            checkpoint, wav_path, recording_dir=recording_dir,
            output_path=output_dir / "analysis_report.json",
            device=str(payload.get("device", "auto")), batch_size=8,
            previous_reports=previous,
        )
        with _jobs_lock:
            _jobs[inference_id].update({"status": "completed", "return_code": 0, "report": report, "finished_at": time.time()})
    except Exception as exc:
        with _jobs_lock:
            _jobs[inference_id].update({"status": "failed", "return_code": 1, "error": str(exc), "finished_at": time.time()})


def start_inference(payload: dict) -> dict:
    training_id = str(payload.get("training_id", ""))
    _training_output(training_id)
    wav_path = payload.get("wav_path")
    recording_dir = payload.get("recording_dir")
    if not isinstance(wav_path, str) or not wav_path.strip():
        raise ValueError("wav_path es obligatorio")
    wav_path = _safe_workspace_path(wav_path, "WAV")
    if not wav_path.is_file() or wav_path.suffix.lower() != ".wav":
        raise ValueError("wav_path debe ser un WAV existente")
    if recording_dir is not None:
        recording_dir = _safe_workspace_path(str(recording_dir), "grabación")
    inference_id = f"inference_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    output_dir = INFERENCE_ROOT / inference_id
    output_dir.mkdir(parents=True, exist_ok=False)
    job = {"inference_id": inference_id, "training_id": training_id, "wav_path": str(wav_path),
           "recording_dir": str(recording_dir or wav_path.parent), "status": "starting",
           "output_dir": str(output_dir), "started_at": time.time()}
    with _jobs_lock:
        _jobs[inference_id] = job
    threading.Thread(target=_inference_worker, args=(inference_id, payload, output_dir), daemon=True).start()
    return job


class Handler(BaseHTTPRequestHandler):
    server_version = "MachiningAssistantMonitor/1.0"

    def log_message(self, format: str, *args) -> None:
        return

    def send_json(self, value: object, status: int = 200) -> None:
        payload = json_bytes(value)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 64 * 1024:
            raise ValueError("request body too large")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                body = (WEB_ROOT / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/status":
                live = None
                service_error = None
                try:
                    live = service_get("/health/live")
                except Exception as exc:
                    # MachiningMonitor is an optional dependency for the
                    # independent RTSP/CNN1D-VAE test-training phase.
                    service_error = str(exc)
                ready = None
                ready_error = None
                if live is not None:
                    try:
                        ready = service_get("/health/ready")
                    except Exception as exc:
                        ready_error = str(exc)
                self.send_json({"service_url": SERVICE_URL, "live": live, "ready": ready, "ready_error": ready_error, "service_error": service_error, "data_root": str(DATA_ROOT), "api_key_configured": bool(API_KEY), "rtsp_training": {"independent": True, "default_url": "rtsp://127.0.0.1:8555/audio", "validation_endpoint": "/api/rtsp-training/validate"}})
                return
            if parsed.path == "/api/machines":
                self.send_json(service_get("/api/recording/machines"))
                return
            if parsed.path == "/api/local-dataset":
                self.send_json(local_dataset_inventory())
                return
            if parsed.path == "/api/storage/health":
                self.send_json(storage_health())
                return
            if parsed.path == "/api/executions":
                machine_id = parse_qs(parsed.query).get("machine_id", [""])[0]
                if not machine_id:
                    raise ValueError("machine_id is required")
                executions = [execution_summary(item) for item in service_get("/api/recording/executions") if item.get("machineId") == machine_id]
                known = {item.get("executionId") for item in executions}
                executions.extend(item for item in local_execution_summaries(machine_id) if item.get("executionId") not in known)
                executions.sort(key=lambda item: str(item.get("captureStartedAtUtc") or ""), reverse=True)
                self.send_json(executions)
                return
            if parsed.path == "/api/execution":
                query = parse_qs(parsed.query)
                machine_id = query.get("machine_id", [""])[0]
                execution_id = safe_execution_id(query.get("execution_id", [""])[0])
                detail = service_get(f"/api/recording/execution/{machine_id}/{execution_id}")
                self.send_json(execution_summary(detail))
                return
            if parsed.path == "/api/training/status":
                training_id = parse_qs(parsed.query).get("training_id", [""])[0]
                with _jobs_lock:
                    job = dict(_jobs.get(training_id, {"training_id": training_id, "status": "unknown"}))
                output_dir = Path(job.get("output_dir", "")) if job.get("output_dir") else None
                if output_dir and output_dir.is_dir():
                    job["artifacts"] = sorted(str(p.relative_to(output_dir)) for p in output_dir.rglob("*") if p.is_file())
                    history = output_dir / "training_history.csv"
                    if history.is_file():
                        job["history"] = history.read_text(encoding="utf-8-sig")[-12000:]
                self.send_json(job)
                return
            if parsed.path == "/api/training/result":
                training_id = parse_qs(parsed.query).get("training_id", [""])[0]
                with _jobs_lock:
                    job = dict(_jobs.get(training_id, {}))
                output_dir = Path(job.get("output_dir", "")) if job.get("output_dir") else None
                if output_dir is None or not output_dir.is_dir():
                    model = training_id.rsplit("_", 1)[-1]
                    job_id = training_id[:-(len(model) + 1)] if model in ("cnn", "lstm") else ""
                    if job_id:
                        output_dir = RUNS_ROOT / job_id / "training" / model
                if output_dir is None or not output_dir.is_dir():
                    raise FileNotFoundError("No existe el directorio del entrenamiento")
                self.send_json(_training_result(output_dir))
                return
            if parsed.path == "/api/training/history":
                try:
                    limit = int(parse_qs(parsed.query).get("limit", ["50"])[0])
                    self.send_json({"ok": True, "runs": storage_training_history(limit)})
                except Exception as exc:
                    self.send_json({"ok": False, "runs": [], "error": str(exc)})
                return
            if parsed.path == "/api/inference/models":
                self.send_json(inference_models())
                return
            if parsed.path == "/api/inference/status":
                inference_id = parse_qs(parsed.query).get("inference_id", [""])[0]
                with _jobs_lock:
                    job = dict(_jobs.get(inference_id, {"inference_id": inference_id, "status": "unknown"}))
                output_dir = Path(job.get("output_dir", "")) if job.get("output_dir") else None
                if output_dir and output_dir.is_dir():
                    report_path = output_dir / "analysis_report.json"
                    if report_path.is_file():
                        job["report"] = _read_json_file(report_path, {})
                self.send_json(job)
                return
            if parsed.path == "/api/rtsp-training/listener/status":
                listener_id = parse_qs(parsed.query).get("listener_id", [""])[0]
                self.send_json(rtsp_listener_status(listener_id))
                return
            if parsed.path == "/api/rtsp-training/listener/active":
                self.send_json(active_rtsp_listener_status())
                return
            if parsed.path == "/api/inference/result":
                inference_id = parse_qs(parsed.query).get("inference_id", [""])[0]
                if not inference_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in inference_id):
                    raise ValueError("inference_id inválido")
                report_path = INFERENCE_ROOT / inference_id / "analysis_report.json"
                if not report_path.is_file():
                    raise FileNotFoundError("No existe el análisis solicitado")
                self.send_json(_read_json_file(report_path, {}))
                return
            self.send_json({"error": "not_found"}, 404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 400)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self.read_json()
            if parsed.path == "/api/import":
                ids = payload.get("execution_ids")
                if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
                    raise ValueError("execution_ids must be a list of strings")
                self.send_json(_persist_import_best_effort(import_recordings(ids)), 201)
                return
            if parsed.path == "/api/import-single":
                execution_id = payload.get("execution_id")
                if not isinstance(execution_id, str) or not execution_id.strip():
                    raise ValueError("execution_id es obligatorio")
                self.send_json(_persist_import_best_effort(import_one_service_recording(execution_id.strip())), 201)
                return
            if parsed.path == "/api/import-local":
                self.send_json(_persist_import_best_effort(import_local_dataset()), 201)
                return
            if parsed.path == "/api/import-local-recording":
                folder = payload.get("folder")
                if not isinstance(folder, str) or not folder.strip():
                    raise ValueError("folder es obligatorio")
                self.send_json(_persist_import_best_effort(import_one_local_recording(folder.strip())), 201)
                return
            if parsed.path == "/api/training/start":
                self.send_json(start_training(payload), 202)
                return
            if parsed.path == "/api/rtsp-training/validate":
                self.send_json(validate_rtsp_training_source(payload), 200)
                return
            if parsed.path == "/api/rtsp-training/listener/start":
                self.send_json(start_rtsp_listener(payload), 202)
                return
            if parsed.path == "/api/rtsp-training/listener/stop":
                self.send_json(stop_rtsp_listener(str(payload.get("listener_id", ""))), 200)
                return
            if parsed.path == "/api/rtsp-training/train":
                self.send_json(request_rtsp_training(payload), 200)
                return
            if parsed.path == "/api/inference/start":
                self.send_json(start_inference(payload), 202)
                return
            self.send_json({"error": "not_found"}, 404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 400)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=setting("MACHINING_WEB_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(setting("MACHINING_WEB_PORT", "8765")))
    args = parser.parse_args()
    if not WEB_ROOT.is_dir() or not (WEB_ROOT / "index.html").is_file():
        raise SystemExit(f"Missing web assets: {WEB_ROOT}")
    print(f"Machining Assistant monitor: http://{args.host}:{args.port}", flush=True)
    print(f"MachiningMonitor API: {SERVICE_URL}; data root: {DATA_ROOT}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
