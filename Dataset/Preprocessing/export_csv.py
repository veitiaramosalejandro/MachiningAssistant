"""Export review copies of existing sample CSV indexes without changing WAV data.

Existing train/val/test CSV labels take precedence over generation-time labels
in samples.json. Output indexes retain paths relative to the source data_dir;
the export folder is a review folder, not a new standalone training dataset.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

if __package__:
    from .active_dataset import LEGACY_DATA_DIR, resolve_dataset_dir
else:
    from active_dataset import LEGACY_DATA_DIR, resolve_dataset_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = LEGACY_DATA_DIR
EXPORT_ROOT = PROJECT_ROOT/'Dataset'/'CSV_exports'
ACTIVE_SPLITS = ('train', 'val', 'test')
ALL_SPLITS = (*ACTIVE_SPLITS, 'exclude')
IDENTITY_FIELDS = ('sample_id', 'base_sample_id', 'recording_id', 'source_recording_id',
                   'split', 'wav_file', 'frame_count')
REQUIRED_FIELDS = (*IDENTITY_FIELDS, 'label', 'label_index')


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    if not headers or any(None in row for row in rows):
        raise ValueError(f'{path}: invalid CSV header or unnamed extra cells')
    return headers, rows


def _class_names(data_dir):
    mapping = _read_json(data_dir/'labels.json')
    if not isinstance(mapping, dict) or len(mapping) < 2 or set(mapping) != {str(i) for i in range(len(mapping))}:
        raise ValueError('labels.json must map consecutive indices 0..N-1, with at least two classes')
    names = []
    for index in range(len(mapping)):
        name = mapping[str(index)]
        if not isinstance(name, str) or not name.strip():
            raise ValueError('labels.json class names must be nonempty strings')
        names.append(name.strip())
    if len(names) != len(set(names)):
        raise ValueError('labels.json class names must be unique')
    return names


def _identity(row):
    missing = set(IDENTITY_FIELDS)-set(row)
    if missing:
        raise ValueError(f'Missing identity fields: {sorted(missing)}')
    result = {}
    for key in IDENTITY_FIELDS:
        value = row[key]
        if value is None or not str(value).strip():
            raise ValueError(f'{key} must be nonempty')
        result[key] = int(value) if key == 'frame_count' else str(value).strip()
    if result['frame_count'] <= 0 or result['split'] not in ALL_SPLITS:
        raise ValueError('frame_count must be positive and split must be train, val, test, or exclude')
    return result


def _write_csv(path, headers, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            converted = {}
            for key in headers:
                value = row.get(key, '')
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
                converted[key] = value
            writer.writerow(converted)


def export_csv_indexes(data_dir, output_dir):
    """Export current labels and frozen sample identities into a new review folder.

    If all three active CSVs are absent, their original rows can be rebuilt
    from samples.json. A partially missing CSV set is rejected rather than
    mixing edited labels with stale generation labels.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f'Output directory is not empty: {output_dir}')
    names = _class_names(data_dir)
    original_rows = _read_json(data_dir/'samples.json')
    if not isinstance(original_rows, list) or not original_rows:
        raise ValueError('samples.json must contain original sample metadata')
    original = {}
    for row in original_rows:
        identity = _identity(row)
        if identity['sample_id'] in original:
            raise ValueError(f"Duplicate generation sample_id: {identity['sample_id']}")
        original[identity['sample_id']] = row
    paths = {split: data_dir/f'{split}.csv' for split in ACTIVE_SPLITS}
    existing = {split: path.is_file() for split, path in paths.items()}
    if any(existing.values()) and not all(existing.values()):
        missing = [paths[split].name for split, found in existing.items() if not found]
        raise FileNotFoundError(f'Incomplete current CSV set: missing {", ".join(missing)}; edited labels were not replaced from samples.json')

    headers, rows, csv_sources = [], [], []
    if all(existing.values()):
        label_authority = 'current_split_csvs'
        selected_paths = [(split, paths[split]) for split in ACTIVE_SPLITS]
        excluded = data_dir/'excluded.csv'
        if excluded.is_file():
            selected_paths.append(('exclude', excluded))
        elif any(row['split'] == 'exclude' for row in original_rows):
            raise FileNotFoundError('excluded.csv is missing while generation metadata contains excluded samples')
        for split, path in selected_paths:
            columns, selected = _read_csv(path)
            for name in columns:
                if name not in headers:
                    headers.append(name)
            if split in ACTIVE_SPLITS and not selected:
                raise ValueError(f'{path.name} must be nonempty')
            for row in selected:
                if str(row.get('split', '')).strip() != split:
                    raise ValueError(f'{path.name}: a row has a different split')
            rows.extend(selected)
            csv_sources.append(str(path))
    else:
        label_authority = 'generation_metadata_no_current_split_csvs'
        rows = [dict(row) for row in original_rows]
        for row in rows:
            for key in row:
                if key not in headers:
                    headers.append(key)
        for split in ACTIVE_SPLITS:
            if not any(row['split'] == split for row in rows):
                raise ValueError(f'Generation metadata has no {split} samples')

    seen, wav_paths = set(), set()
    source_splits, base_splits = defaultdict(set), defaultdict(set)
    label_differences = 0
    counts = {split: Counter() for split in ALL_SPLITS}
    for row in rows:
        if set(REQUIRED_FIELDS)-set(row):
            raise ValueError(f'Missing sample fields: {sorted(set(REQUIRED_FIELDS)-set(row))}')
        identity = _identity(row)
        sample_id = identity['sample_id']
        if sample_id in seen:
            raise ValueError(f'Duplicate current sample_id: {sample_id}')
        seen.add(sample_id)
        if sample_id not in original or identity != _identity(original[sample_id]):
            raise ValueError(f'{sample_id}: current source/base/path/frame/split identity differs from samples.json')
        label_index = int(row['label_index'])
        label = str(row['label']).strip()
        if not 0 <= label_index < len(names) or label != names[label_index]:
            raise ValueError(f'{sample_id}: current label/index does not match labels.json')
        row.update(identity)
        row['label'] = label
        row['label_index'] = label_index
        relative = Path(identity['wav_file'])
        if relative.is_absolute():
            raise ValueError(f'{sample_id}: wav_file must be relative to source data_dir')
        wav = (data_dir/relative).resolve()
        if not wav.is_file():
            raise FileNotFoundError(f'Indexed WAV not found: {wav}')
        if wav in wav_paths:
            raise ValueError(f'Duplicate indexed WAV path: {wav}')
        wav_paths.add(wav)
        if identity['split'] in ACTIVE_SPLITS:
            source_splits[identity['source_recording_id']].add(identity['split'])
            base_splits[identity['base_sample_id']].add(identity['split'])
        counts[identity['split']][label] += 1
        if (label != str(original[sample_id].get('label', '')).strip()
                or label_index != int(original[sample_id].get('label_index', -1))):
            label_differences += 1
    if seen != set(original):
        raise ValueError('Current CSV membership differs from samples.json; no samples were silently dropped or reassigned')
    if any(len(value) > 1 for value in source_splits.values()):
        raise ValueError('A source_recording_id crosses training/validation/test splits')
    if any(len(value) > 1 for value in base_splits.values()):
        raise ValueError('A base_sample_id crosses training/validation/test splits')

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ALL_SPLITS:
        name = 'excluded.csv' if split == 'exclude' else f'{split}.csv'
        _write_csv(output_dir/name, headers, [row for row in rows if row['split'] == split])
    _write_csv(output_dir/'samples.csv', headers, rows)
    summary = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_data_dir': str(data_dir), 'output_dir': str(output_dir),
        'sample_count': len(rows), 'split_counts': {split: sum(counts[split].values()) for split in ALL_SPLITS},
        'class_names': names, 'label_mapping': {str(i): name for i, name in enumerate(names)},
        'split_label_counts': {split: {name: counts[split][name] for name in names} for split in ALL_SPLITS},
        'label_authority': label_authority, 'generation_label_differences_preserved': label_differences,
        'input_csv_files': csv_sources,
        'generation_metadata_file': str(data_dir/'samples.json'),
        'labels_file': str(data_dir/'labels.json'),
        'relative_path_base': str(data_dir), 'export_directory_is_standalone_dataset': False,
        'verification': {'source_groups_disjoint': True, 'base_groups_disjoint': True,
                         'sample_ids_unique': True, 'wav_paths_unique': True,
                         'source_split_assignments_preserved': True,
                         'source_csvs_modified': False, 'wav_files_modified_or_generated': False},
    }
    (output_dir/'export_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    readme = f'''CSV index export for review
===========================

Source dataset: {data_dir}
Exported samples: {len(rows)}
Label authority: {label_authority}
Generation-label differences retained: {label_differences}

train.csv, val.csv, test.csv, excluded.csv and samples.csv preserve sample membership,
source-recording groups and the current labels. Existing split CSV labels take precedence
over generation-time labels in samples.json. labels.json supplies the current class mapping.
No WAV was cut, copied, regenerated or modified. No split was randomized or reassigned.

IMPORTANT PATH CONTRACT
-----------------------
wav_file and other relative sample paths still resolve against the source dataset above,
not this export directory. These files are review copies, not a standalone dataset.
Do not pass this export folder to the trainer's --data-dir argument.
After review, you may manually place the desired CSVs in the original dataset directory
when you intend to update its indexes. This exporter never writes back to that directory.

Reproduce from the project root
-------------------------------
python Dataset/Preprocessing/export_csv.py --data-dir /path/to/existing_dataset
Optional --output-dir selects a new empty review directory. Otherwise a Lisbon timestamped
directory is created under Dataset/CSV_exports/.
If train.csv, val.csv and test.csv are all absent, samples.json can supply the initial rows.
A partially missing active CSV set is rejected so edited labels cannot be silently replaced.
'''
    (output_dir/'README.txt').write_text(readme, encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=None,
                        help='Prepared dataset; default follows current_dataset.json, then the legacy directory')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    output = args.output_dir
    if output is None:
        stamp = datetime.now(ZoneInfo('Europe/Lisbon')).strftime('%Y-%m-%d_%H-%M-%S_%f')
        output = EXPORT_ROOT/stamp
    summary = export_csv_indexes(resolve_dataset_dir(args.data_dir), output)
    print(json.dumps({'output_dir': summary['output_dir'], 'sample_count': summary['sample_count'],
                      'split_counts': summary['split_counts'], 'label_authority': summary['label_authority']}, indent=2))


if __name__ == '__main__':
    main()
