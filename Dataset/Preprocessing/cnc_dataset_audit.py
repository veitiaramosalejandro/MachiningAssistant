"""Audit and prepare service-provided audio/context/label samples.

The service boundary is intentionally in-memory: an external application sends
one complete audio window, one aligned MachiningContext snapshot and one
validated label. This module does not read local context files and does not
invent missing values. It produces a report plus raw, model-ready records; the
existing audio and CNC normalizers remain separate training concerns.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
import math

import numpy as np

from .cnc_features import build_feature_vector, feature_schema


DEFAULT_SAMPLE_RATE = 44100
DEFAULT_VALID_FRAMES = 220500
CHANNELS = 2
SPLITS = ("train", "val", "test")

_REQUIRED_RECORD_FIELDS = (
    "sample_id",
    "recording_id",
    "source_recording_id",
    "split",
    "sample_rate",
    "start_frame",
    "end_frame_exclusive",
    "label",
    "label_index",
    "context",
)


def _reason(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _as_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if isinstance(value, str) and str(result) != value.strip():
        raise ValueError(f"{name} must be an integer")
    return result


def _finite_float(name: str, value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _pcm_from_record(record: Mapping[str, object], valid_frames: int) -> np.ndarray:
    if "pcm" in record and record["pcm"] is not None:
        pcm = np.asarray(record["pcm"], dtype=np.float32)
    elif "pcm16_le" in record:
        payload = record["pcm16_le"]
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise ValueError("pcm16_le must be bytes-like")
        raw = bytes(payload)
        expected_bytes = valid_frames * CHANNELS * 2
        if len(raw) != expected_bytes:
            raise ValueError(f"pcm16_le must contain exactly {expected_bytes} bytes")
        pcm = np.frombuffer(raw, dtype="<i2").reshape(valid_frames, CHANNELS).astype(np.float32)
    else:
        raise ValueError("record must contain pcm or pcm16_le")

    if pcm.shape != (valid_frames, CHANNELS):
        raise ValueError(f"audio must have shape ({valid_frames}, {CHANNELS})")
    if not np.isfinite(pcm).all():
        raise ValueError("audio contains non-finite values")
    return np.ascontiguousarray(pcm, dtype=np.float32)


def _validate_record(
    record: Mapping[str, object],
    class_names: tuple[str, ...],
    sample_rate: int,
    valid_frames: int,
    source_splits: Mapping[str, set[str]],
) -> tuple[dict, list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    if not isinstance(record, Mapping):
        return {}, [_reason("record_not_object", "record must be a mapping")]

    missing = [name for name in _REQUIRED_RECORD_FIELDS if name not in record]
    if missing:
        errors.append(_reason("missing_record_fields", ", ".join(sorted(missing))))
        return {}, errors

    prepared = dict(record)
    for name in ("sample_id", "recording_id", "source_recording_id", "split", "label"):
        if not isinstance(record[name], str) or not record[name].strip():
            errors.append(_reason("invalid_text_field", f"{name} must be a non-empty string"))

    split = record.get("split")
    if split not in SPLITS:
        errors.append(_reason("invalid_split", f"split must be one of {SPLITS}"))
    elif len(source_splits.get(str(record.get("source_recording_id")), set())) > 1:
        errors.append(_reason("source_split_leakage", "source_recording_id appears in multiple splits"))

    try:
        received_rate = _as_int("sample_rate", record["sample_rate"])
        if received_rate != sample_rate:
            errors.append(_reason("sample_rate_mismatch", f"expected {sample_rate}, received {received_rate}"))
        start = _as_int("start_frame", record["start_frame"])
        end = _as_int("end_frame_exclusive", record["end_frame_exclusive"])
        if start < 0 or end - start != valid_frames:
            errors.append(_reason("invalid_window", f"window must contain exactly {valid_frames} frames"))
    except ValueError as exc:
        errors.append(_reason("invalid_window_metadata", str(exc)))

    label_index = None
    try:
        label_index = _as_int("label_index", record["label_index"])
        if not 0 <= label_index < len(class_names):
            errors.append(_reason("label_index_out_of_range", str(label_index)))
        elif record.get("label") != class_names[label_index]:
            errors.append(_reason("label_mismatch", "label does not match class_names[label_index]"))
    except ValueError as exc:
        errors.append(_reason("invalid_label_index", str(exc)))

    try:
        pcm = _pcm_from_record(record, valid_frames)
        prepared["pcm"] = pcm
    except (TypeError, ValueError) as exc:
        errors.append(_reason("invalid_audio", str(exc)))

    try:
        prepared["cnc_features"] = build_feature_vector(record["context"])
    except (TypeError, ValueError) as exc:
        errors.append(_reason("invalid_context", str(exc)))

    if label_index is not None:
        prepared["label_index"] = label_index
    return prepared, errors


def audit_window_records(
    records: Iterable[Mapping[str, object]],
    class_names: Iterable[str],
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    valid_frames: int = DEFAULT_VALID_FRAMES,
) -> dict:
    """Audit service records and return a reproducible data-quality report.

    Records with errors are excluded from ``accepted_records``. The report is
    not ready for training unless every supplied record is valid and no source
    recording crosses train/validation/test boundaries.
    """
    names = tuple(class_names)
    if len(names) < 2 or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("class_names must contain at least two non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("class_names must be unique")

    materialized = list(records)
    source_splits: dict[str, set[str]] = defaultdict(set)
    for record in materialized:
        if isinstance(record, Mapping):
            source = record.get("source_recording_id")
            split = record.get("split")
            if source is not None and split in SPLITS:
                source_splits[str(source)].add(str(split))

    accepted: list[dict] = []
    rejected: list[dict] = []
    for index, record in enumerate(materialized):
        prepared, errors = _validate_record(record, names, sample_rate, valid_frames, source_splits)
        if errors:
            rejected.append({
                "index": index,
                "sample_id": record.get("sample_id") if isinstance(record, Mapping) else None,
                "errors": errors,
            })
        else:
            accepted.append(prepared)

    report = {
        "schema_version": 1,
        "sample_rate": sample_rate,
        "valid_frames": valid_frames,
        "channels": CHANNELS,
        "class_names": list(names),
        "feature_schema": feature_schema(),
        "input_contract": "audio_pcm_plus_aligned_machining_context_plus_one_label",
        "source_split_map": {key: sorted(value) for key, value in sorted(source_splits.items())},
        "counts": {
            "received": len(materialized),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "labels": dict(Counter(item["label"] for item in accepted)),
            "splits": dict(Counter(item["split"] for item in accepted)),
        },
        "rejected_records": rejected,
        "ready_for_training": bool(materialized) and not rejected,
    }
    return {"report": report, "accepted_records": accepted}


__all__ = [
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_VALID_FRAMES",
    "CHANNELS",
    "SPLITS",
    "audit_window_records",
]
