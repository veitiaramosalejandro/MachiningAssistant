"""Versioned, context-aware reconstruction-score baselines for VAE runs."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import unicodedata

import numpy as np


BASELINE_VERSION = "phase3-baseline-v1"
DIMENSIONS = ("machine", "operation", "tool", "material", "program", "context_type")
MIN_GLOBAL_SAMPLES = 20
MIN_CONTEXT_SAMPLES = 20


def _text(value) -> str | None:
    if value is None:
        return None
    result = unicodedata.normalize("NFC", str(value)).strip()
    return result or None


def _finite(values) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def score_statistics(values) -> dict:
    array = _finite(values)
    if not len(array):
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None,
                "p90": None, "p95": None, "p99": None}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def _context_value(context: dict, metadata: dict, dimension: str):
    tool = context.get("Tool") if isinstance(context.get("Tool"), dict) else {}
    values = {
        "machine": context.get("MachineName") or context.get("Machine") or metadata.get("MachineName"),
        "operation": context.get("Operation"),
        "tool": tool.get("Tool") or tool.get("name") if isinstance(tool, dict) else tool,
        "material": context.get("PartMaterial") or context.get("Material"),
        "program": context.get("ProgramName") or context.get("Work") or metadata.get("ProgramName"),
        "context_type": context.get("MachineState") or context.get("State") or context.get("Operation"),
    }
    return _text(values.get(dimension))


def _recording_documents(data_dir: Path):
    folders = {
        unicodedata.normalize("NFC", path.name): path
        for path in data_dir.iterdir() if path.is_dir()
    }
    cache = {}

    def get(recording_id: str):
        key = unicodedata.normalize("NFC", str(recording_id))
        if key in cache:
            return cache[key]
        folder = folders.get(key, data_dir / recording_id)
        try:
            metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            metadata = {}
        try:
            document = json.loads((folder / "contexts.json").read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            document = {}
        contexts = document.get("contexts", []) if isinstance(document, dict) else []
        try:
            event_document = json.loads((folder / "events.json").read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            event_document = {}
        events = event_document.get("annotations", []) if isinstance(event_document, dict) else []
        cache[key] = (metadata if isinstance(metadata, dict) else {},
                      contexts if isinstance(contexts, list) else [],
                      events if isinstance(events, list) else [])
        return cache[key]

    return get


def _context_at(contexts: list, timestamp_s: float) -> dict:
    selected = None
    for item in contexts:
        if not isinstance(item, dict):
            continue
        try:
            timestamp = float(item.get("timestamp_seconds", 0) or 0)
        except (TypeError, ValueError):
            continue
        if timestamp <= timestamp_s:
            selected = item
        else:
            break
    value = selected.get("context", {}) if isinstance(selected, dict) else {}
    return value if isinstance(value, dict) else {}


def _explicit_label(row: dict, events: list) -> bool | None:
    """Read only explicit anomaly labels; quality scores are not inferred as labels."""
    for key in ("is_anomaly", "anomaly"):
        if isinstance(row.get(key), bool):
            return row[key]
    for event in events:
        if not isinstance(event, dict):
            continue
        try:
            timestamp = float(event.get("timestamp_seconds", 0) or 0)
            start, end = float(row.get("start_s", 0)), float(row.get("end_s", 0))
        except (TypeError, ValueError):
            continue
        if not start <= timestamp <= end:
            continue
        candidates = [event.get("label"), event.get("classification"),
                      (event.get("driver") or {}).get("Classification") if isinstance(event.get("driver"), dict) else None]
        for candidate in candidates:
            value = _text(candidate)
            if value is None:
                continue
            normalized = value.casefold()
            if normalized in {"normal", "ok", "good", "pass", "healthy"}:
                return False
            if normalized in {"anomaly", "anomalous", "fault", "failure", "bad", "defect"}:
                return True
    return None


def _calibrate_validation(train_scores, validation_rows: list[dict]) -> dict:
    train = _finite(train_scores)
    labeled = []
    for row in validation_rows:
        try:
            score = float(row.get("score"))
        except (TypeError, ValueError):
            continue
        if row.get("label") is not None and np.isfinite(score):
            labeled.append((score, bool(row["label"])))
    thresholds = score_statistics(train)
    result = {
        "status": "validation_unlabeled",
        "method": "train_percentiles_with_validation_monitoring",
        "calibrated_threshold": None,
        "validation_count": len(validation_rows),
        "labeled_validation_count": len(labeled),
        "validation_above_train_p95_rate": None,
        "validation_above_train_p99_rate": None,
        "metrics": None,
    }
    if len(validation_rows) and thresholds["p95"] is not None:
        scores = _finite(row["score"] for row in validation_rows)
        result["validation_above_train_p95_rate"] = float(np.mean(scores > thresholds["p95"])) if len(scores) else None
        result["validation_above_train_p99_rate"] = float(np.mean(scores > thresholds["p99"])) if len(scores) else None
    if not labeled:
        return result
    candidates = sorted({score for score, _ in labeled})
    best = None
    for threshold in candidates:
        predicted = [score > threshold for score, _ in labeled]
        actual = [label for _, label in labeled]
        tp = sum(p and a for p, a in zip(predicted, actual))
        fp = sum(p and not a for p, a in zip(predicted, actual))
        fn = sum(not p and a for p, a in zip(predicted, actual))
        tn = sum(not p and not a for p, a in zip(predicted, actual))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidate = (f1, recall, -threshold, threshold, tp, fp, fn, tn, precision)
        if best is None or candidate > best:
            best = candidate
    _, recall, _, threshold, tp, fp, fn, tn, precision = best
    result.update({
        "status": "label_calibrated", "method": "validation_f1_threshold",
        "calibrated_threshold": float(threshold),
        "metrics": {
            "precision": float(precision), "recall": float(recall),
            "f1": float(best[0]), "accuracy": float((tp + tn) / len(labeled)),
            "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        },
    })
    return result


def build_baseline_report(rows: list[dict], data_dir: str | Path, *, min_context_samples: int = MIN_CONTEXT_SAMPLES) -> dict:
    """Build global/context baselines and classify train/validation/test windows."""
    get_documents = _recording_documents(Path(data_dir).expanduser().resolve())
    enriched = []
    for row in rows:
        value = row.get("best_mse_standardized", row.get("trained_mse_standardized"))
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = math.nan
        metadata, contexts, events = get_documents(str(row.get("recording_id", "")))
        center = (float(row.get("start_s", 0)) + float(row.get("end_s", 0))) / 2
        context = _context_at(contexts, center)
        dimensions = {dimension: _context_value(context, metadata, dimension) for dimension in DIMENSIONS}
        enriched.append({**row, "score": score, "dimensions": dimensions,
                         "label": _explicit_label(row, events)})

    train_rows = [row for row in enriched if row.get("split") == "train" and np.isfinite(row["score"])]
    validation_rows = [row for row in enriched if row.get("split") == "val" and np.isfinite(row["score"])]
    global_stats = score_statistics(row["score"] for row in train_rows)
    context_stats = {dimension: {} for dimension in DIMENSIONS}
    for dimension in DIMENSIONS:
        groups = defaultdict(list)
        for row in train_rows:
            value = row["dimensions"].get(dimension)
            if value is not None:
                groups[value].append(row["score"])
        context_stats[dimension] = {
            value: score_statistics(scores) for value, scores in sorted(groups.items())
        }

    calibration = _calibrate_validation([row["score"] for row in train_rows], validation_rows)
    if global_stats["count"] < MIN_GLOBAL_SAMPLES:
        readiness = "insufficient_data"
    else:
        readiness = "ready"
    threshold = calibration.get("calibrated_threshold") or global_stats.get("p99")
    for row in enriched:
        stats = global_stats
        source = "global"
        operation = row["dimensions"].get("operation")
        if operation and operation in context_stats["operation"] and context_stats["operation"][operation]["count"] >= min_context_samples:
            stats = context_stats["operation"][operation]
            source = "operation"
        if not np.isfinite(row["score"]):
            row.update({"state": "insufficient_data", "threshold_source": source, "threshold": None})
            continue
        context_known = any(row["dimensions"].values())
        if readiness != "ready":
            state = "insufficient_data"
        elif stats.get("p95") is None:
            state = "insufficient_data"
        elif source == "global" and context_known and row["score"] > global_stats.get("p99", math.inf):
            state = "out_of_distribution"
        elif row["score"] <= stats["p95"]:
            state = "normal"
        elif row["score"] <= (threshold if source == "global" else stats.get("p99")):
            state = "review"
        else:
            state = "anomalous"
        row.update({"state": state, "threshold_source": source,
                    "threshold": threshold if source == "global" else stats.get("p99")})

    serializable_rows = [
        {key: value for key, value in row.items() if key not in {"dimensions", "label"}}
        | {"context_dimensions": row["dimensions"], "explicit_label": row["label"]}
        for row in enriched
    ]
    return {
        "schema_version": 1,
        "baseline_version": BASELINE_VERSION,
        "status": readiness,
        "source": "train_reconstruction_scores",
        "threshold_policy": "p95=review, p99=anomalous; labeled validation may replace the global anomaly threshold",
        "minimum_global_samples": MIN_GLOBAL_SAMPLES,
        "minimum_context_samples": min_context_samples,
        "global": global_stats,
        "by_context": context_stats,
        "validation_calibration": calibration,
        "window_count": len(enriched),
        "window_states": serializable_rows,
        "limitations": [
            "Without explicit validation labels, thresholds are statistical references, not calibrated fault probabilities.",
            "Quality scores are not automatically interpreted as anomaly labels.",
            "Context baselines require enough training windows; otherwise the global baseline is used.",
        ],
    }
