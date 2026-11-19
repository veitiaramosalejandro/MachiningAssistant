"""Independent, persisted inference for a new recording with a VAE checkpoint."""
from __future__ import annotations

from collections import Counter
import importlib
import json
import math
from pathlib import Path
import sys
import uuid

import numpy as np
import torch


INFERENCE_SCHEMA_VERSION = 1
MIN_BASELINE_SAMPLES = 20
ANOMALOUS_STATES = {"anomalous", "out_of_distribution"}


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return default


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def _load_runtime(model_family: str):
    directory_name = "CNN1D-VAE" if model_family == "CNN1D-VAE" else "LSTM-VAE"
    directory = Path(__file__).resolve().parent / directory_name
    if not directory.is_dir():
        raise FileNotFoundError(f"No existe el runtime para {model_family}")
    # CNN1D-VAE and LSTM-VAE expose modules with the same names (runtime,
    # model and data). Remove only those aliases so switching architectures in
    # the same web process cannot silently reuse the previous model family.
    for name in ("runtime", "model", "data"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(directory))
    try:
        module = importlib.import_module("runtime")
    finally:
        sys.path.pop(0)
    return module


def _context_at(document: dict, timestamp_s: float) -> dict:
    entries = document.get("contexts", []) if isinstance(document, dict) else []
    selected = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            if float(entry.get("timestamp_seconds", 0) or 0) <= timestamp_s:
                selected = entry
            else:
                break
        except (TypeError, ValueError):
            continue
    context = selected.get("context", {}) if isinstance(selected, dict) else {}
    return context if isinstance(context, dict) else {}


def _recording_documents(recording_dir: Path):
    metadata = _read_json(recording_dir / "metadata.json", {})
    contexts = _read_json(recording_dir / "contexts.json", {})
    events_doc = _read_json(recording_dir / "events.json", {})
    playback = _read_json(recording_dir / "playback.json", {})
    events = events_doc.get("annotations", []) if isinstance(events_doc, dict) else []
    return (metadata if isinstance(metadata, dict) else {},
            contexts if isinstance(contexts, dict) else {},
            events if isinstance(events, list) else [],
            playback if isinstance(playback, dict) else {})


def _context_fields(context: dict) -> dict:
    tool = context.get("Tool", {})
    tool_name = tool.get("name") or tool.get("Tool") if isinstance(tool, dict) else tool
    parameters = context.get("Parameters", {})
    return {
        "machine": context.get("MachineName") or context.get("Machine"),
        "operation": context.get("Operation"),
        "tool": tool_name,
        "material": context.get("PartMaterial") or context.get("Material"),
        "program": context.get("ProgramName") or context.get("Work"),
        "context_type": context.get("MachineState") or context.get("State") or context.get("Operation"),
        "parameters": parameters if isinstance(parameters, dict) else {},
    }


def _overlap_events(events: list, start_s: float, end_s: float) -> list:
    result = []
    for event in events:
        if not isinstance(event, dict):
            continue
        timestamp = _finite(event.get("timestamp_seconds"))
        if timestamp is None or not start_s <= timestamp <= end_s:
            continue
        result.append({
            "timestamp_seconds": timestamp,
            "quality": event.get("quality"),
            "machine_state": event.get("machine_state_label") or event.get("machine_state"),
            "observations": event.get("observations"),
            "label": event.get("label") or event.get("classification"),
        })
    return result


def _state_for_score(score: float, baseline: dict, context_fields: dict) -> tuple[str, float | None, str, float | None]:
    global_stats = baseline.get("global", {}) if isinstance(baseline, dict) else {}
    by_context = baseline.get("by_context", {}) if isinstance(baseline, dict) else {}
    operation = context_fields.get("operation")
    operation_stats = by_context.get("operation", {}).get(operation) if operation else None
    if not isinstance(operation_stats, dict) or operation_stats.get("count", 0) < baseline.get("minimum_context_samples", 20):
        operation_stats = None
    stats = operation_stats or global_stats
    source = "operation" if operation_stats else "global"
    minimum_samples = baseline.get("minimum_context_samples", 20) if operation_stats else baseline.get("minimum_global_samples", MIN_BASELINE_SAMPLES)
    if not stats or stats.get("count", 0) < minimum_samples:
        return "insufficient_data", None, source, None
    p95, p99 = _finite(stats.get("p95")), _finite(stats.get("p99"))
    if p95 is None or p99 is None:
        return "insufficient_data", None, source, None
    threshold = _finite(baseline.get("validation_calibration", {}).get("calibrated_threshold"))
    if source == "global" and not operation_stats and operation:
        state = "out_of_distribution" if score > p99 else ("review" if score > p95 else "normal")
    elif score <= p95:
        state = "normal"
    elif score <= (threshold if source == "global" and threshold is not None else p99):
        state = "review"
    else:
        state = "anomalous"
    limit = threshold if source == "global" and threshold is not None else p99
    if state == "normal":
        confidence = _clamp((p95 - score) / max(abs(p95), 1e-12))
    elif state == "review":
        confidence = _clamp(0.5 + (score - p95) / max(abs(limit - p95), 1e-12) * 0.35)
    elif state == "anomalous":
        confidence = _clamp(0.65 + (score - limit) / max(abs(limit), 1e-12) * 0.35)
    else:
        confidence = 0.0
    return state, limit, source, confidence


def _incidents(rows: list[dict]) -> list[dict]:
    incidents = []
    current = []
    for row in rows + [{"state": "__end__"}]:
        if row.get("state") in ANOMALOUS_STATES:
            current.append(row)
            continue
        if not current:
            continue
        events = [event for item in current for event in item.get("events", [])]
        contexts = Counter(item.get("context", {}).get("operation") or "Sin contexto" for item in current)
        incidents.append({
            "incident_id": f"incident-{len(incidents) + 1:04d}",
            "start_s": current[0]["start_s"], "end_s": current[-1]["end_s"],
            "duration_s": current[-1]["end_s"] - current[0]["start_s"],
            "window_count": len(current),
            "max_score": max(item["score"] for item in current),
            "states": sorted(set(item["state"] for item in current)),
            "operations": [item[0] for item in contexts.most_common()],
            "tools": sorted({item.get("context", {}).get("tool") for item in current if item.get("context", {}).get("tool")}),
            "parameters": current[0].get("context", {}).get("parameters", {}),
            "events": events,
        })
        current = []
    return incidents


def _comparison(report: dict, previous_reports: list[Path] | None) -> list[dict]:
    result = []
    for path in previous_reports or []:
        previous = _read_json(path, {})
        if not isinstance(previous, dict) or not previous.get("summary"):
            continue
        old = previous["summary"]
        result.append({
            "analysis_id": previous.get("analysis_id"), "source": str(path),
            "mean_mse_delta": report["summary"]["mse_mean"] - old.get("mse_mean", report["summary"]["mse_mean"]),
            "anomalous_window_delta": report["summary"]["anomalous_windows"] - old.get("anomalous_windows", 0),
            "anomalous_duration_delta_s": report["summary"]["anomalous_duration_s"] - old.get("anomalous_duration_s", 0),
        })
    return result


def analyze_recording(checkpoint: str | Path, wav_path: str | Path, *, recording_dir: str | Path | None = None,
                      output_path: str | Path | None = None, device: str = "auto", batch_size: int = 8,
                      previous_reports: list[Path] | None = None) -> dict:
    """Analyze one WAV without training and optionally persist a complete report."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    wav_path = Path(wav_path).expanduser().resolve()
    if not checkpoint.is_file() or not wav_path.is_file():
        raise FileNotFoundError("Checkpoint y WAV deben existir")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    family = state.get("model_family")
    if family not in {"CNN1D-VAE", "LSTM-VAE"}:
        raise ValueError("El checkpoint no es un checkpoint VAE compatible")
    runtime = _load_runtime(family)
    engine = runtime.VAEEngine.from_checkpoint(checkpoint, device=device)
    prediction = runtime.predict_wav(engine, wav_path, max_windows=0, batch_size=batch_size)
    baseline = _read_json(checkpoint.parent / "baseline_report.json", {})
    recording_dir = Path(recording_dir).expanduser().resolve() if recording_dir else wav_path.parent
    metadata, context_doc, events, playback = _recording_documents(recording_dir)
    offset_s = (_finite(playback.get("AudioOffsetMs")) or 0.0) / 1000.0
    rows = []
    for item in prediction.get("windows", []):
        start_s, end_s = float(item["start_s"]), float(item["end_s"])
        context = _context_fields(_context_at(context_doc, (start_s + end_s) / 2 + offset_s))
        score = float(item["reconstruction_mse_standardized"])
        state_name, threshold, threshold_source, confidence = _state_for_score(score, baseline, context)
        rows.append({**item, "score": score, "state": state_name, "threshold": threshold,
                     "threshold_source": threshold_source, "confidence": confidence,
                     "confidence_calibration_status": baseline.get("validation_calibration", {}).get("status", "unavailable"),
                     "confidence_type": "statistical_baseline_confidence_not_fault_probability",
                     "context": context, "events": _overlap_events(events, start_s, end_s)})
    scores = [row["score"] for row in rows if math.isfinite(row["score"])]
    anomalous = [row for row in rows if row["state"] in ANOMALOUS_STATES]
    incidents = _incidents(rows)
    summary = {
        "mse_global": float(np.mean(scores)) if scores else None,
        "mse_mean": float(np.mean(scores)) if scores else None,
        "mse_max": float(np.max(scores)) if scores else None,
        "window_count": len(rows), "anomalous_windows": len(anomalous),
        "anomalous_window_percent": (len(anomalous) / len(rows) * 100) if rows else 0.0,
        "anomalous_duration_s": float(sum(row["end_s"] - row["start_s"] for row in anomalous)),
        "first_anomaly_s": anomalous[0]["start_s"] if anomalous else None,
        "last_anomaly_s": anomalous[-1]["end_s"] if anomalous else None,
        "incident_count": len(incidents),
        "state_counts": dict(Counter(row["state"] for row in rows)),
    }
    report = {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        "analysis_id": f"analysis-{uuid.uuid4().hex[:12]}",
        "status": "completed", "model_family": family,
        "checkpoint": str(checkpoint), "checkpoint_baseline_version": baseline.get("baseline_version"),
        "wav_file": str(wav_path), "recording_dir": str(recording_dir),
        "sample_rate_hz": prediction.get("sample_rate_hz"), "window_duration_s": rows[0].get("window_duration_s") if rows else None,
        "model_status": prediction.get("model_status"), "device": prediction.get("device"),
        "confidence_calibration_status": baseline.get("validation_calibration", {}).get("status", "unavailable"),
        "summary": summary, "incidents": incidents, "windows": rows,
        "comparison": _comparison({"summary": summary}, previous_reports),
        "limitations": [
            "La confianza es estadística y no representa probabilidad calibrada de fallo.",
            "Una grabación nueva no modifica ni reentrena el checkpoint.",
            "La detección depende de la representatividad del baseline del entrenamiento.",
        ],
    }
    if output_path:
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report
