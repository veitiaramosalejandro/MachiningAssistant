"""Read editable recording-level train/validation/test assignments from CSV."""
from __future__ import annotations

import csv
from pathlib import Path
import unicodedata


SPLIT_NAMES = ("train", "val", "test")
CSV_FIELDS = ("recording_id", "source_recording_id", "split")


def normalize_identifier(value: str) -> str:
    """Use one Unicode form for CSV identifiers and filesystem folder names."""
    return unicodedata.normalize("NFC", str(value).strip())


def read_split_csv(path: str | Path) -> list[dict[str, str]]:
    """Read the current file on every call and reject source-recording leakage.

    ``exclude`` records remain in the table but do not enter any data split.
    Names refer to sample directories, rather than individual signal windows.
    """
    path = Path(path)
    rows = []
    seen = set()
    source_assignments = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(CSV_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing CSV columns: {', '.join(sorted(missing))}")
        for line, entry in enumerate(reader, start=2):
            row = {key: (entry.get(key) or "").strip() for key in CSV_FIELDS}
            if not any(row.values()):
                continue
            if not all(row.values()):
                raise ValueError(f"{path}:{line}: recording_id, source_recording_id and split must be filled")
            recording_id = row["recording_id"]
            if recording_id in seen:
                raise ValueError(f"{path}:{line}: duplicate recording_id: {recording_id}")
            seen.add(recording_id)
            split = row["split"]
            if split not in (*SPLIT_NAMES, "exclude"):
                raise ValueError(f"{path}:{line}: split must be train, val, test or exclude; got {split!r}")
            if split != "exclude":
                source_id = row["source_recording_id"]
                previous = source_assignments.setdefault(source_id, split)
                if previous != split:
                    raise ValueError(
                        f"{path}:{line}: source recording {source_id!r} appears in both {previous} and {split}"
                    )
            rows.append(row)
    empty = [split for split in SPLIT_NAMES if not any(row["split"] == split for row in rows)]
    if empty:
        raise ValueError(f"{path}: empty data splits: {', '.join(empty)}")
    return rows


def split_recording_ids(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    return {split: [row["recording_id"] for row in rows if row["split"] == split] for split in SPLIT_NAMES}


def write_split_csv(path: str | Path, rows: list[dict[str, str]]) -> None:
    """Save the resolved assignments actually used by a run."""
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in CSV_FIELDS} for row in rows)
