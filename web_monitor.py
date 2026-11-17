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
from urllib.parse import parse_qs, urlparse
import uuid
import wave


PROJECT_ROOT = Path(__file__).resolve().parent
WEB_ROOT = PROJECT_ROOT / "web_monitor"
RUNS_ROOT = PROJECT_ROOT / "runs" / "monitor_training"
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
LOCAL_DATA_ROOT = PROJECT_ROOT / "Dataset" / "Data"

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


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
            rows.append({"recording_id": recording_id, "source_recording_id": execution_id, "split": split})
            imported.append({"execution_id": execution_id, "recording_id": recording_id, "wav": str(target_wav), "wav_info": wav_info(target_wav)})
        split_csv = root / "dataset_splits.csv"
        write_split_csv(split_csv, rows)
        manifest = {"job_id": job_id, "sample_rate_hz": sample_rate, "data_dir": str(data_dir), "split_csv": str(split_csv), "recordings": imported}
        (root / "import_manifest.json").write_bytes(json_bytes(manifest))
        return manifest
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


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


def _read_json_file(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return default


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
                    score = float(row.get("trained_mse_standardized", "nan"))
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
                    "operation_context": _context_at(context_doc, metadata, context_time_s),
                    "events": overlap_events,
                    "playback_offset_ms": playback_offset_ms,
                    "source_start_seconds": (metadata.get("SourceRecording") or {}).get("StartSeconds") if isinstance(metadata.get("SourceRecording"), dict) else None,
                    "recording_start": metadata.get("StartTime"),
                }
                rows.append(enriched)
    finite_rows = [row for row in rows if math.isfinite(row["score"])]
    train_scores = sorted(row["score"] for row in finite_rows if row["split"] == "train")
    threshold = train_scores[min(len(train_scores) - 1, int(len(train_scores) * 0.99))] if train_scores else None
    for row in finite_rows:
        row["above_train_p99"] = threshold is not None and row["score"] > threshold
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
        "history": report.get("history", []),
        "baseline": report.get("mean_reconstruction_baseline_mse", {}),
        "best_epoch": report.get("best_epoch"),
        "best_checkpoint": report.get("best_checkpoint"),
        "score_reference_train_p99": threshold,
        "score_reference_note": "Referencia estadística del train; no es un umbral operativo calibrado.",
        "context_counts": sorted(context_counts.values(), key=lambda item: item["max_score"], reverse=True),
        "top_windows": top_windows,
        "window_count": len(finite_rows),
    }


def start_training(payload: dict) -> dict:
    job_id = str(payload.get("import_job_id", ""))
    if not job_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in job_id):
        raise ValueError("import_job_id inválido")
    model = str(payload.get("model", "cnn")).lower()
    if model not in {"cnn", "lstm"}:
        raise ValueError("model debe ser cnn o lstm")
    allowed = {
        "epochs": {20, 40, 60, 100, 120, 200, 350},
        "windows_per_recording": {16, 64, 128, 300, 600},
        "sequence_length": {2048, 8192, 16384},
        "latent_dim": {8, 16, 32},
        "batch_size": {8, 16, 32},
        "learning_rate": {0.0003, 0.001},
        "beta": {0.0001, 0.001, 0.01},
        "eval_every": {5, 10, 20},
        "device": {"auto", "cuda", "cpu"},
    }
    def selected(name: str, default, converter):
        value = converter(payload.get(name, default))
        if value not in allowed[name]:
            raise ValueError(f"{name} no está entre las opciones permitidas")
        return value

    epochs = selected("epochs", 20, int)
    windows = selected("windows_per_recording", 64, int)
    sequence_length = selected("sequence_length", 2048, int)
    latent_dim = selected("latent_dim", 16, int)
    batch_size = selected("batch_size", 8, int)
    learning_rate = selected("learning_rate", 0.001, float)
    beta = selected("beta", 0.001, float)
    eval_every = selected("eval_every", 10, int)
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
        "--output-dir", str(output_dir), "--epochs", str(epochs), "--windows-per-recording", str(windows),
        "--sequence-length", str(sequence_length), "--latent-dim", str(latent_dim),
        "--batch-size", str(batch_size), "--learning-rate", str(learning_rate), "--beta", str(beta),
        "--eval-every", str(eval_every), "--device", device,
    ]
    training_id = f"{job_id}_{model}"
    job = {"training_id": training_id, "import_job_id": job_id, "model": model, "status": "starting", "args": args[1:], "configuration": {"epochs": epochs, "windows_per_recording": windows, "sequence_length": sequence_length, "latent_dim": latent_dim, "batch_size": batch_size, "learning_rate": learning_rate, "beta": beta, "eval_every": eval_every, "device": device}, "started_at": time.time(), "output_dir": str(output_dir), "log_tail": ""}
    with _jobs_lock:
        _jobs[training_id] = job
    threading.Thread(target=train_worker, args=(training_id, args, output_dir), daemon=True).start()
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
                live = service_get("/health/live")
                ready = None
                ready_error = None
                try:
                    ready = service_get("/health/ready")
                except Exception as exc:
                    ready_error = str(exc)
                self.send_json({"service_url": SERVICE_URL, "live": live, "ready": ready, "ready_error": ready_error, "data_root": str(DATA_ROOT), "api_key_configured": bool(API_KEY)})
                return
            if parsed.path == "/api/machines":
                self.send_json(service_get("/api/recording/machines"))
                return
            if parsed.path == "/api/local-dataset":
                self.send_json(local_dataset_inventory())
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
                self.send_json(import_recordings(ids), 201)
                return
            if parsed.path == "/api/import-local":
                self.send_json(import_local_dataset(), 201)
                return
            if parsed.path == "/api/training/start":
                self.send_json(start_training(payload), 202)
                return
            self.send_json({"error": "not_found"}, 404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 400)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("MACHINING_WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MACHINING_WEB_PORT", "8765")))
    args = parser.parse_args()
    if not WEB_ROOT.is_dir() or not (WEB_ROOT / "index.html").is_file():
        raise SystemExit(f"Missing web assets: {WEB_ROOT}")
    print(f"Machining Assistant monitor: http://{args.host}:{args.port}", flush=True)
    print(f"MachiningMonitor API: {SERVICE_URL}; data root: {DATA_ROOT}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
