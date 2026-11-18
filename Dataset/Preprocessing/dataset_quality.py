"""Quality validation for raw recordings and their time-aligned JSON documents."""
from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import wave

import numpy as np


JSON_COLLECTIONS = {
    "contexts.json": "contexts",
    "events.json": "annotations",
}
OPTIONAL_JSON = ("contexts.json", "events.json", "playback.json")
NON_NEGATIVE_KEYS = {
    "programmedrpm", "adjustedrpm", "actualspindlerpm", "spindlespeed",
    "programmedmmpermin", "adjustedmmpermin", "actualfeedmmpermin", "feedrate",
    "stkxy", "stkz", "incxy", "incz", "heightmm", "diametermm", "cornerradiusmm",
    "startseconds", "endseconds",
    "flutes", "usagetimeatprogramstartminutes", "usagepercent", "powerratio",
}
PERCENT_KEYS = {"adjustmentpercent", "remainingusepercent"}


def _issue(code: str, message: str, severity: str = "error", path: str | None = None) -> dict:
    value = {"code": code, "message": message, "severity": severity}
    if path:
        value["path"] = path
    return value


def _read_json(path: Path) -> tuple[object | None, dict | None]:
    if not path.is_file():
        return None, _issue("missing_json", f"Missing JSON document: {path.name}", "warning", str(path))
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")), None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, _issue("invalid_json", f"Invalid JSON in {path.name}: {exc}", "error", str(path))


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _timestamp(value: object, path: str, duration_s: float, issues: list[dict]) -> None:
    if not _finite_number(value):
        issues.append(_issue("invalid_timestamp", "Timestamp must be a finite number", "error", path))
        return
    seconds = float(value)
    if seconds < -0.5 or seconds > duration_s + 0.5:
        issues.append(_issue(
            "timestamp_out_of_range",
            f"Timestamp {seconds:.3f}s is outside the audio duration {duration_s:.3f}s",
            "error", path,
        ))


def _validate_numeric_context(value: object, path: str, issues: list[dict]) -> None:
    if "parameter_changes" in path:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            _validate_numeric_context(child, child_path, issues)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_numeric_context(child, f"{path}[{index}]", issues)
        return
    key = path.rsplit(".", 1)[-1].split("[", 1)[0].lower()
    if key not in NON_NEGATIVE_KEYS and key not in PERCENT_KEYS:
        return
    if value is None:
        return
    if not _finite_number(value):
        issues.append(_issue("invalid_cnc_value", "CNC numeric value must be finite", "error", path))
        return
    number = float(value)
    if key in NON_NEGATIVE_KEYS and number < 0:
        issues.append(_issue("negative_cnc_value", "CNC value cannot be negative", "error", path))
    if key in PERCENT_KEYS and not 0 <= number <= 200:
        issues.append(_issue("invalid_percentage", "Percentage/offset value is outside the accepted range 0..200", "error", path))


def _metadata_duration(metadata: dict, duration_s: float) -> tuple[float, list[dict]]:
    issues: list[dict] = []
    expected = duration_s
    specialized = metadata.get("SpecializedSourceRecording")
    if isinstance(specialized, dict):
        start = specialized.get("start_seconds")
        end = specialized.get("end_seconds")
        if _finite_number(start) and _finite_number(end) and float(end) >= float(start):
            expected = float(end) - float(start)
        else:
            issues.append(_issue("invalid_source_interval", "Specialized source interval is invalid", "error", "SpecializedSourceRecording"))
    declared = metadata.get("DurationSeconds")
    duration_delta = abs(float(declared) - expected) if _finite_number(declared) else 0.0
    if _finite_number(declared) and duration_delta > max(1.0, expected * 0.02):
        # Service segment metadata can retain the original full-session duration.
        if not isinstance(specialized, dict):
            issues.append(_issue(
                "duration_mismatch",
                f"Metadata duration {float(declared):.3f}s differs from audio duration {expected:.3f}s",
                "error" if duration_delta > max(5.0, expected * 0.10) else "warning", "DurationSeconds",
            ))
    for key in ("StartTime", "EndTime"):
        if key in metadata:
            try:
                datetime.fromisoformat(str(metadata[key]).replace("Z", "+00:00"))
            except (TypeError, ValueError):
                issues.append(_issue("invalid_datetime", f"{key} is not a valid ISO-8601 datetime", "error", key))
    return expected, issues


def validate_recording(
    directory: str | Path,
    recording_id: str,
    source_recording_id: str,
    split: str,
    *,
    metadata: dict | None = None,
    pcm: np.ndarray | None = None,
    sample_rate: int | None = None,
) -> dict:
    """Validate one recording after its WAV has been decoded by the loader."""
    directory = Path(directory)
    issues: list[dict] = []
    wav_path = directory / "audio_000000.wav"
    if pcm is None:
        try:
            with wave.open(str(wav_path), "rb") as handle:
                sample_rate = handle.getframerate()
                channels = handle.getnchannels()
                width = handle.getsampwidth()
                frames = handle.getnframes()
                payload = handle.readframes(frames)
            if len(payload) != frames * channels * width:
                issues.append(_issue("truncated_wav", "WAV payload is shorter than its declared frame count", "error", str(wav_path)))
            pcm = np.frombuffer(payload, dtype="<i2").reshape(-1, channels).astype(np.float32)
        except (OSError, EOFError, ValueError, wave.Error) as exc:
            issues.append(_issue("invalid_wav", f"Unable to read WAV: {exc}", "error", str(wav_path)))
            pcm = np.empty((0, 2), dtype=np.float32)
            sample_rate = sample_rate or 0
            channels = 0
            width = 0
    else:
        pcm = np.asarray(pcm)
        channels = pcm.shape[-1] if pcm.ndim == 2 else 0
        width = 2
        frames = len(pcm) if pcm.ndim == 2 else 0

    if sample_rate != 44100 or channels != 2 or width != 2:
        issues.append(_issue(
            "unsupported_wav_format",
            f"Expected 44100 Hz stereo PCM16, got {sample_rate} Hz, {channels} channels, {width * 8} bits",
            "error", str(wav_path),
        ))
    if pcm.ndim != 2 or pcm.shape[-1] != 2 or not np.isfinite(pcm).all():
        issues.append(_issue("non_finite_audio", "Decoded audio must be a finite two-channel array", "error", str(wav_path)))
        frames = 0
        duration_s = 0.0
    else:
        frames = len(pcm)
        duration_s = frames / float(sample_rate or 1)
        clipped = int(np.count_nonzero(np.abs(pcm) >= 32767))
        clipping_ratio = clipped / max(1, pcm.size)
        if clipping_ratio > 0:
            issues.append(_issue(
                "audio_clipping",
                f"{clipped} PCM samples ({clipping_ratio:.6%}) are at the clipping boundary",
                "warning", str(wav_path),
            ))

    if metadata is None:
        metadata, error = _read_json(directory / "metadata.json")
        if error:
            issues.append(error)
    if not isinstance(metadata, dict):
        issues.append(_issue("invalid_metadata", "metadata.json must contain a JSON object", "error", "metadata.json"))
        metadata = {}

    identifiers = {
        str(metadata.get(key)).strip()
        for key in ("ExecutionId", "RecordingId", "execution_id", "recording_id")
        if metadata.get(key) is not None
    }
    specialized = metadata.get("SpecializedSourceRecording")
    if isinstance(specialized, dict) and specialized.get("execution_id"):
        identifiers.add(str(specialized["execution_id"]).strip())
    if identifiers and recording_id not in identifiers and source_recording_id not in identifiers:
        issues.append(_issue(
            "recording_id_mismatch",
            f"CSV identifiers ({recording_id}, {source_recording_id}) do not match metadata identifiers {sorted(identifiers)}",
            "error", "metadata.ExecutionId",
        ))
    source_recording = metadata.get("SourceRecording")
    if isinstance(source_recording, dict) and source_recording.get("Folder"):
        source_folder = str(source_recording["Folder"]).replace("/", "\\").rstrip("\\").split("\\")[-1]
        if source_folder and source_folder != source_recording_id:
            issues.append(_issue(
                "source_recording_mismatch",
                f"Metadata SourceRecording folder {source_folder!r} differs from CSV source {source_recording_id!r}",
                "error", "metadata.SourceRecording.Folder",
            ))

    _, duration_issues = _metadata_duration(metadata, duration_s)
    issues.extend(duration_issues)
    _validate_numeric_context(metadata, "metadata", issues)

    documents: dict[str, object] = {}
    for filename in OPTIONAL_JSON:
        document, error = _read_json(directory / filename)
        if error:
            issues.append(error)
        else:
            documents[filename] = document
    for filename, collection in JSON_COLLECTIONS.items():
        document = documents.get(filename)
        if document is None:
            continue
        if not isinstance(document, dict) or not isinstance(document.get(collection), list):
            issues.append(_issue("invalid_collection", f"{filename} must contain a '{collection}' array", "error", filename))
            continue
        for index, entry in enumerate(document[collection]):
            if not isinstance(entry, dict):
                issues.append(_issue("invalid_entry", "Collection entries must be JSON objects", "error", f"{filename}.{collection}[{index}]"))
                continue
            _timestamp(entry.get("timestamp_seconds"), f"{filename}.{collection}[{index}].timestamp_seconds", duration_s, issues)
            _validate_numeric_context(entry, f"{filename}.{collection}[{index}]", issues)
        time_reference = document.get("time_reference")
        if time_reference not in (None, "playback_seconds"):
            issues.append(_issue("unsupported_time_reference", f"Unsupported time reference: {time_reference!r}", "error", filename))

    # Cross-check the optional context/event identifiers when both documents exist.
    contexts = documents.get("contexts.json")
    events = documents.get("events.json")
    if isinstance(contexts, dict) and isinstance(events, dict):
        context_ids = {entry.get("id") for entry in contexts.get("contexts", []) if isinstance(entry, dict)}
        for index, event in enumerate(events.get("annotations", [])):
            driver = event.get("driver") if isinstance(event, dict) else None
            context_id = driver.get("ContextId") if isinstance(driver, dict) else None
            if context_id and context_ids and context_id not in context_ids:
                issues.append(_issue("unknown_context_id", f"Event references unknown context id {context_id!r}", "warning", f"events.json.annotations[{index}]"))

    errors = [item for item in issues if item["severity"] == "error"]
    warnings = [item for item in issues if item["severity"] == "warning"]
    return {
        "recording_id": recording_id,
        "source_recording_id": source_recording_id,
        "split": split,
        "status": "invalid" if errors else "warning" if warnings else "valid",
        "duration_s": duration_s,
        "frame_count": frames,
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "bit_depth": width * 8,
        "json_presence": {name: (directory / name).is_file() for name in ("metadata.json", *OPTIONAL_JSON)},
        "errors": errors,
        "warnings": warnings,
    }


def summarize_quality(records: list[dict], ignored_recordings: list[str] | None = None) -> dict:
    """Create a serializable dataset quality report from per-recording results."""
    counts = {status: sum(record["status"] == status for record in records) for status in ("valid", "warning", "invalid")}
    split_counts = {split: sum(record["split"] == split for record in records) for split in ("train", "val", "test")}
    return {
        "schema_version": 1,
        "valid": counts["invalid"] == 0,
        "ready_for_training": counts["invalid"] == 0 and all(split_counts[split] > 0 for split in split_counts),
        "recording_count": len(records),
        "status_counts": counts,
        "split_counts": split_counts,
        "ignored_recordings": sorted(ignored_recordings or []),
        "records": records,
    }
