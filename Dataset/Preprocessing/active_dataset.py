"""Resolve and publish the active prepared dataset using only the standard library.

Explicit data directories retain normal caller-relative path semantics. Paths
stored in current_dataset.json are either absolute or relative to PROJECT_ROOT,
regardless of the pointer file's location. Importing this module writes nothing.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_DATA_DIR = PROJECT_ROOT/'Dataset'/'Data'/'fourclass_5s'
ACTIVE_DATASET_POINTER = Path(__file__).resolve().parent/'current_dataset.json'
REQUIRED_CSV_COLUMNS = (
    'sample_id', 'base_sample_id', 'recording_id', 'source_recording_id',
    'split', 'label', 'label_index', 'wav_file', 'frame_count',
    'start_frame', 'end_frame_exclusive', 'start_seconds',
    'end_seconds_exclusive', 'duration_seconds',
)


def resolve_dataset_dir(data_dir=None, pointer_path=None) -> Path:
    """Resolve an explicit directory, the current pointer, or the legacy fallback.

    No pointer is consulted for an explicit path. A missing pointer falls back
    to the legacy directory; malformed existing pointers are reported as errors.
    Dataset contents are validated by their caller, not read during resolution.
    """
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()
    pointer = Path(pointer_path).expanduser().resolve() if pointer_path is not None else ACTIVE_DATASET_POINTER
    if not pointer.exists():
        return LEGACY_DATA_DIR.resolve()
    if not pointer.is_file():
        raise ValueError(f'Active dataset pointer is not a file: {pointer}')
    value = json.loads(pointer.read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict) or type(value.get('schema_version')) is not int or value['schema_version'] != 1:
        raise ValueError(f'{pointer}: expected schema_version 1')
    directory = value.get('data_dir')
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError(f'{pointer}: data_dir must be a nonempty path string')
    target = Path(directory).expanduser()
    if not target.is_absolute():
        target = PROJECT_ROOT/target
    return target.resolve()


def _check_prepared_headers(data_dir: Path) -> None:
    if not data_dir.is_dir():
        raise FileNotFoundError(f'Prepared dataset directory not found: {data_dir}')
    labels_path = data_dir/'labels.json'
    if not labels_path.is_file():
        raise FileNotFoundError(f'Prepared dataset is missing labels.json: {data_dir}')
    mapping = json.loads(labels_path.read_text(encoding='utf-8-sig'))
    if (not isinstance(mapping, dict) or len(mapping) < 2
            or set(mapping) != {str(index) for index in range(len(mapping))}):
        raise ValueError(f'{labels_path}: expected consecutive class indices 0..N-1, N >= 2')
    names = list(mapping.values())
    if (any(not isinstance(name, str) or not name.strip() for name in names)
            or len({name.strip() for name in names}) != len(names)):
        raise ValueError(f'{labels_path}: class names must be nonempty and unique')
    for split in ('train', 'val', 'test'):
        path = data_dir/f'{split}.csv'
        if not path.is_file():
            raise FileNotFoundError(f'Prepared dataset is missing {path.name}: {data_dir}')
        with path.open(encoding='utf-8-sig', newline='') as handle:
            fields = next(csv.reader(handle), [])
        missing = set(REQUIRED_CSV_COLUMNS)-set(fields)
        if missing:
            raise ValueError(f'{path}: missing CSV columns: {", ".join(sorted(missing))}')
        if len(fields) != len(set(fields)):
            raise ValueError(f'{path}: duplicate CSV columns')


def set_active_dataset(data_dir, pointer_path=None) -> Path:
    """Check prepared-file headers and atomically publish their directory.

    This does not load waveform values, import ML libraries, modify data files,
    or claim that complete sample contents have been validated. The returned
    path is the resolved dataset directory.
    """
    target = Path(data_dir).expanduser().resolve()
    _check_prepared_headers(target)
    pointer = Path(pointer_path).expanduser().resolve() if pointer_path is not None else ACTIVE_DATASET_POINTER
    try:
        stored_path = target.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        stored_path = str(target)
    value = {'schema_version': 1, 'data_dir': stored_path,
             'updated_at_utc': datetime.now(timezone.utc).isoformat()}
    pointer.parent.mkdir(parents=True, exist_ok=True)
    temporary = pointer.with_name(pointer.name+'.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        temporary.replace(pointer)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


publish_active_dataset = set_active_dataset
