"""Validate recording/audio/context integrity before model training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Dataset.Preprocessing.company_data import DEFAULT_SOURCE, DEFAULT_SPLIT_CSV, read_wav
from Dataset.Preprocessing.dataset_quality import summarize_quality, validate_recording
from Dataset.Preprocessing.dataset_splits import SPLIT_NAMES, normalize_identifier, read_split_csv


def validate_dataset(source: str | Path = DEFAULT_SOURCE, split_csv: str | Path = DEFAULT_SPLIT_CSV) -> dict:
    source = Path(source).expanduser().resolve()
    split_path = Path(split_csv).expanduser().resolve()
    rows = read_split_csv(split_path)
    active = [
        {**row, "recording_id": normalize_identifier(row["recording_id"]),
         "source_recording_id": normalize_identifier(row["source_recording_id"])}
        for row in rows if row["split"] in SPLIT_NAMES
    ]
    found = {
        normalize_identifier(path.name): path
        for path in source.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    records = []
    for row in active:
        directory = found.get(row["recording_id"], source / row["recording_id"])
        metadata_path = directory / "metadata.json"
        wav_path = directory / "audio_000000.wav"
        if not directory.is_dir() or not metadata_path.is_file() or not wav_path.is_file():
            records.append({
                "recording_id": row["recording_id"],
                "source_recording_id": row["source_recording_id"],
                "split": row["split"],
                "status": "invalid",
                "errors": [{"code": "missing_recording_files", "message": "Recording directory, metadata.json and audio_000000.wav are required", "severity": "error"}],
                "warnings": [],
            })
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        sample_rate, pcm = read_wav(wav_path)
        records.append(validate_recording(
            directory, row["recording_id"], row["source_recording_id"], row["split"],
            metadata=metadata, pcm=pcm, sample_rate=sample_rate,
        ))
    report = summarize_quality(records, sorted(set(found) - {row["recording_id"] for row in active}))
    report.update({"source": str(source), "split_csv": str(split_path)})
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--report", type=Path, help="Write the JSON quality report to this path")
    args = parser.parse_args(argv)
    report = validate_dataset(args.data_dir, args.split_csv)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
