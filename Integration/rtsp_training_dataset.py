"""Build a reproducible multimodal classification dataset from one RTSP session."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import shutil

from Dataset.Preprocessing.cnc_features import build_feature_vector
from Integration.stream_quality import QUALITY_LABELS


def _read_json_lines(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fields = ["sample_id", "split", "wav_file", "label", "label_index", "frame_count"]
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_rtsp_dataset(*, session_dir: Path, aligned_path: Path, output_dir: Path) -> dict:
    """Materialize aligned WAV/context/label records for the existing trainer.

    A sample is accepted only when its WAV and all CNC features required by
    ``cnc_features.FEATURE_NAMES`` are present. No missing feature is filled
    with zero or inferred from the classifier prediction.
    """
    session_dir = session_dir.expanduser().resolve()
    aligned_path = aligned_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    rows = _read_json_lines(aligned_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    accepted = []
    rejected = []
    for index, row in enumerate(rows):
        label = row.get("label")
        if label not in QUALITY_LABELS:
            rejected.append({"index": index, "reason": "invalid_label", "label": label})
            continue
        audio_wav_file = row.get("audio_wav_file")
        if not isinstance(audio_wav_file, str) or not audio_wav_file.strip():
            rejected.append({"index": index, "reason": "audio_window_missing"})
            continue
        source = Path(audio_wav_file).expanduser()
        if not source.is_absolute():
            source = (session_dir / source).resolve()
        if not source.is_file():
            rejected.append({"index": index, "reason": "audio_window_missing", "path": str(source)})
            continue
        context = row.get("context")
        if not isinstance(context, dict):
            rejected.append({"index": index, "reason": "cnc_context_missing"})
            continue
        try:
            build_feature_vector(context)
        except (TypeError, ValueError) as exc:
            rejected.append({"index": index, "reason": "cnc_context_invalid", "error": str(exc)})
            continue
        sample_id = f"stream_{index:06d}"
        target = audio_dir / f"{sample_id}.wav"
        shutil.copy2(source, target)
        accepted.append({
            "sample_id": sample_id,
            "label": label,
            "label_index": QUALITY_LABELS.index(label),
            "wav_file": str(Path("audio") / target.name),
            "frame_count": 220500,
            "window_start_seconds": row.get("window_start_seconds"),
            "window_end_seconds": row.get("window_end_seconds"),
            "context": context,
            "quality_score": row.get("quality_score"),
        })

    by_label = defaultdict(list)
    for item in accepted:
        by_label[item["label"]].append(item)
    sample_rows = []
    context_records = []
    split_counts = Counter()
    class_counts = {}
    for label in QUALITY_LABELS:
        items = by_label[label]
        class_counts[label] = len(items)
        # Keep one validation and one test sample per class when possible;
        # all remaining samples are training data. This deliberately refuses
        # to claim readiness when a class cannot populate all three splits.
        for position, item in enumerate(items):
            split = "train"
            if len(items) >= 3 and position == 0:
                split = "val"
            elif len(items) >= 3 and position == 1:
                split = "test"
            split_counts[split] += 1
            sample_rows.append({key: item[key] for key in ("sample_id", "wav_file", "label", "label_index", "frame_count")}
                               | {"split": split})
            context_records.append({"sample_id": item["sample_id"], "context": item["context"]})

    labels = {str(index): label for index, label in enumerate(QUALITY_LABELS)}
    _write_json(output_dir / "labels.json", labels)
    _write_csv(output_dir / "samples.csv", sample_rows)
    _write_json(output_dir / "contexts.json", context_records)
    report = {
        "ready": bool(accepted) and all(class_counts[label] >= 3 for label in QUALITY_LABELS)
        and all(split_counts[split] > 0 for split in ("train", "val", "test")),
        "dataset_dir": str(output_dir),
        "samples_csv": str(output_dir / "samples.csv"),
        "contexts_json": str(output_dir / "contexts.json"),
        "labels_json": str(output_dir / "labels.json"),
        "accepted_samples": len(accepted),
        "rejected_samples": len(rejected),
        "class_counts": class_counts,
        "split_counts": dict(split_counts),
        "rejections": rejected[-100:],
        "required_classes": list(QUALITY_LABELS),
        "label_source": "x-machining-metadata.event.quality",
    }
    _write_json(output_dir / "dataset_report.json", report)
    return report


__all__ = ["build_rtsp_dataset"]
