"""Prepare five-second WAVs and one samples.csv outside the raw Data folder.

The default output is Dataset/Processed_5s and is replaced after a successful
build. Raw recordings are retained. Training remains a separate manual step.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
import csv
from datetime import datetime, timezone
import json
import io
from pathlib import Path
import sys
import wave
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Dataset.Preprocessing.active_dataset import ACTIVE_DATASET_POINTER, resolve_dataset_dir, set_active_dataset
from Dataset.Preprocessing.build_segments import build_dataset_segments
from Dataset.Preprocessing.build_fourclass import LABELS, build_fourclass_dataset, _validated_config
from Dataset.Preprocessing.dataset_splits import read_split_csv, write_split_csv
from Dataset.Preprocessing.recording_inventory import scan_recordings, plan_assignments

DEFAULT_DATA_DIR = PROJECT_ROOT/'Dataset'/'Data'
DEFAULT_SPLIT_CSV = Path(__file__).resolve().parent/'dataset_splits.csv'
DEFAULT_RECIPE = Path(__file__).resolve().parent/'fourclass_config.json'
SPLITS = ('train', 'val', 'test', 'exclude')


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def _bytes_if_present(path):
    return path.read_bytes() if path.exists() else None


def _write_bytes_atomic(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_bytes(value)
    temporary.replace(path)


def _sample_key(row):
    return (row['recording_id'], int(row['start_frame']), int(row['end_frame_exclusive']), row['label'])


def _manual_overrides(active_dir):
    """Keep explicit CSV edits within the selected four-class label definition."""
    if not active_dir.exists():
        return {}
    expected = {str(index): name for index, name in enumerate(LABELS)}
    if _json(active_dir/'labels.json') != expected:
        raise ValueError('The active dataset uses custom class definitions. This preprocessing recipe is synthetic four-class; select the matching label workflow before rebuilding.')
    from Dataset.Preprocessing.fourclass_data import load_manifest
    load_manifest(active_dir)
    original = {row['sample_id']: row for row in _json(active_dir/'samples.json')}
    result = {}
    for split in SPLITS:
        filename = 'excluded.csv' if split == 'exclude' else f'{split}.csv'
        path = active_dir/filename
        if not path.exists() and split == 'exclude':
            continue
        with path.open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                generated = original[row['sample_id']]
                index = int(row['label_index'])
                if not 0 <= index < len(LABELS) or row['label'] != LABELS[index]:
                    raise ValueError(f"{path}: label/index does not match the four-class mapping")
                if row['label'] != generated['label'] or index != int(generated['label_index']):
                    result[_sample_key(generated)] = {
                        'label': row['label'], 'label_index': index,
                        'base_wav': str((active_dir/generated['base_wav']).resolve()),
                    }
    return result


def _pcm_identity(path):
    with wave.open(str(path), 'rb') as handle:
        header = (handle.getframerate(), handle.getnchannels(), handle.getsampwidth(), handle.getnframes())
        return header, handle.readframes(handle.getnframes())


def _preserve_overrides(dataset_dir, overrides):
    if not overrides:
        return 0
    generated = {row['sample_id']: row for row in _json(dataset_dir/'samples.json')}
    known_keys = {_sample_key(row) for row in generated.values()}
    missing = set(overrides)-known_keys
    if missing:
        raise ValueError(f'Manually labelled segments no longer exist in the new build; review those labels first: {sorted(missing)[:3]}')
    compared = set()
    combined = []
    fields = None
    applied = 0
    for split in SPLITS:
        path = dataset_dir/('excluded.csv' if split == 'exclude' else f'{split}.csv')
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames)
            rows = list(reader)
        for row in rows:
            source = generated[row['sample_id']]
            override = overrides.get(_sample_key(source))
            if override is None:
                continue
            old_base = Path(override['base_wav'])
            new_base = (dataset_dir/source['base_wav']).resolve()
            pair = (str(old_base), str(new_base))
            if pair not in compared:
                if _pcm_identity(old_base) != _pcm_identity(new_base):
                    raise ValueError(f"Original PCM changed for a manually labelled segment: {row['sample_id']}; review the label before activating")
                compared.add(pair)
            row['label'] = override['label']
            row['label_index'] = override['label_index']
            row['label_origin'] = 'manual_csv_override_preserved'
            applied += 1
        with path.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        combined.extend(rows)
    with (dataset_dir/'samples.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(combined)
    summary = _json(dataset_dir/'dataset_summary.json')
    summary['generation_label_counts'] = summary['label_counts']
    summary['generation_split_label_counts'] = summary['split_label_counts']
    summary['label_counts'] = dict(Counter(row['label'] for row in combined))
    summary['split_label_counts'] = {
        split: {label: sum(row['split'] == split and row['label'] == label for row in combined) for label in LABELS}
        for split in SPLITS
    }
    summary['manual_csv_overrides_preserved'] = applied
    _atomic_json(dataset_dir/'dataset_summary.json', summary)
    with (dataset_dir/'README.txt').open('a', encoding='utf-8') as handle:
        handle.write(f'\nManual CSV label overrides retained: {applied}. Current split CSVs contain training targets; samples.json retains generation-time labels.\n')
    return applied


def _prepare_versioned_dataset(data_dir=DEFAULT_DATA_DIR, split_csv=DEFAULT_SPLIT_CSV,
                              versions_dir=None, pointer_path=None, recipe_path=DEFAULT_RECIPE,
                              new_source_split='train', dry_run=False, force=False):
    data_dir = Path(data_dir).expanduser().resolve()
    split_csv = Path(split_csv).expanduser().resolve()
    versions_dir = Path(versions_dir).expanduser().resolve() if versions_dir is not None else data_dir/'prepared_versions'
    pointer_path = Path(pointer_path).expanduser().resolve() if pointer_path is not None else ACTIVE_DATASET_POINTER
    recipe_path = Path(recipe_path).expanduser().resolve()
    split_before = split_csv.read_bytes()
    pointer_before = _bytes_if_present(pointer_path)
    recipe_before = recipe_path.read_bytes()
    recipe = _json(recipe_path)
    validated = _validated_config(recipe)
    if validated['segment_frames'] != 220500:
        raise ValueError('This training workflow requires complete 5-second samples (220500 frames)')
    records = scan_recordings(data_dir)
    plan = plan_assignments(records, read_split_csv(split_csv), new_source_split)
    active_dir = resolve_dataset_dir(pointer_path=pointer_path) if pointer_path.exists() else data_dir/'fourclass_5s'
    overrides = _manual_overrides(active_dir)
    signature = {
        'source_data_dir': str(data_dir),
        'recordings': [{'recording_id': row['recording_id'], 'source_recording_id': row['source_recording_id'],
                        'input_signature': row['input_signature']} for row in records],
        'assignments': plan['assignments'], 'recipe': recipe,
        'manual_overrides': [{'key': list(key), 'label': value['label'], 'label_index': value['label_index']}
                             for key, value in sorted(overrides.items())],
    }
    frame_counts = {row['recording_id']: row['frame_count'] for row in records}
    base_counts = {split: sum(frame_counts[row['recording_id']]//220500 for row in plan['assignments'] if row['split'] == split)
                   for split in SPLITS}
    report = {'schema_version': 1, 'label_mode': 'synthetic_fourclass',
              'raw_data_dir': str(data_dir), 'recording_count': len(records),
              'new_recordings': plan['new_recordings'], 'assignments': plan['assignments'],
              'source_groups': plan['source_groups'], 'new_source_split': new_source_split,
              'base_split_counts': base_counts, 'split_counts': {key: value*4 for key, value in base_counts.items()},
              'short_tail_count': sum(row['frame_count'] % 220500 != 0 for row in records),
              'manual_override_count': len(overrides), 'previous_active_data_dir': str(active_dir),
              'active_pointer': str(pointer_path), 'label_assumption': 'Original recordings are good by the selected synthetic-demo convention, not inferred from event quality scores.'}
    old_report_path = active_dir.parent/'preparation.json'
    previous = _json(old_report_path) if old_report_path.is_file() else None
    unchanged = (previous is not None and previous.get('status') == 'prepared'
                 and previous.get('input_signature') == signature)
    if dry_run:
        return {**report, 'status': 'dry_run', 'would_rebuild': force or not unchanged,
                'data_dir': str(active_dir), 'version_dir': str(active_dir.parent) if previous else None}
    if unchanged and not force:
        from Dataset.Preprocessing.fourclass_data import load_manifest
        load_manifest(active_dir)
        return {**report, 'status': 'unchanged', 'data_dir': str(active_dir), 'version_dir': str(active_dir.parent)}
    stamp = datetime.now(ZoneInfo('Europe/Lisbon')).strftime('%Y-%m-%d_%H-%M-%S_%f')
    version = versions_dir/stamp
    version.mkdir(parents=True, exist_ok=False)
    prepared_dir = version/'fourclass_5s'
    report.update(status='building', version_dir=str(version), data_dir=str(prepared_dir),
                  created_at_utc=datetime.now(timezone.utc).isoformat(), input_signature=signature)
    _atomic_json(version/'preparation.json', report)
    snapshot = version/'recording_splits.csv'
    write_split_csv(snapshot, plan['assignments'])
    _atomic_json(version/'recipe_used.json', recipe)
    try:
        build_dataset_segments(data_dir, snapshot, version/'segments_5s', segment_seconds=5.)
        build_fourclass_dataset(version/'segments_5s', prepared_dir, recipe)
        applied = _preserve_overrides(prepared_dir, overrides)
        from Dataset.Preprocessing.fourclass_data import load_manifest
        manifest = load_manifest(prepared_dir)
        if len(manifest) != sum(report['split_counts'][split] for split in ('train', 'val', 'test')):
            raise ValueError('Prepared sample counts differ from the recording plan')
        after = scan_recordings(data_dir)
        after_signatures = [{'recording_id': row['recording_id'], 'source_recording_id': row['source_recording_id'],
                             'input_signature': row['input_signature']} for row in after]
        if after_signatures != signature['recordings']:
            raise RuntimeError('Raw recordings changed during preprocessing; rerun after copying is finished')
        current_overrides = _manual_overrides(active_dir)
        current_override_signature = [
            {'key': list(key), 'label': value['label'], 'label_index': value['label_index']}
            for key, value in sorted(current_overrides.items())
        ]
        if current_override_signature != signature['manual_overrides']:
            raise RuntimeError('Active CSV labels changed during preprocessing; rerun with the latest labels')
        if split_csv.read_bytes() != split_before or _bytes_if_present(pointer_path) != pointer_before or recipe_path.read_bytes() != recipe_before:
            raise RuntimeError('Split table, recipe or active pointer changed during preprocessing; rerun with the latest inputs')
        report.update(status='prepared', manual_override_count=applied,
                      reference_rms_pcm=_json(prepared_dir/'resolved_config.json')['reference_rms_pcm'])
        _atomic_json(version/'preparation.json', report)
        # Publish only the completed build. Restore both small control files if
        # the final pointer write fails; source WAVs and old datasets are untouched.
        try:
            _write_bytes_atomic(split_csv, snapshot.read_bytes())
            set_active_dataset(prepared_dir, pointer_path=pointer_path)
        except Exception:
            _write_bytes_atomic(split_csv, split_before)
            if pointer_before is None:
                if pointer_path.exists():
                    pointer_path.unlink()
            else:
                _write_bytes_atomic(pointer_path, pointer_before)
            raise
        return report
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', active_dataset_changed=False)
        _atomic_json(version/'preparation.json', report)
        raise


def prepare_dataset(data_dir=DEFAULT_DATA_DIR, split_csv=DEFAULT_SPLIT_CSV,
                    versions_dir=None, pointer_path=None, recipe_path=DEFAULT_RECIPE,
                    new_source_split='train', dry_run=False, force=False, *, output_dir=None):
    """Use a fixed processed folder; retain explicitly requested legacy versions."""
    if versions_dir is not None:
        if output_dir is not None:
            raise ValueError('Choose either output_dir or versions_dir, not both')
        return _prepare_versioned_dataset(data_dir, split_csv, versions_dir, pointer_path,
                                          recipe_path, new_source_split, dry_run, force)
    from Dataset.Preprocessing.fixed_preparation import prepare_fixed_dataset
    raw_root = Path(data_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve() if output_dir is not None else raw_root.parent/'Processed_5s'
    pointer = Path(pointer_path).expanduser().resolve() if pointer_path is not None else ACTIVE_DATASET_POINTER
    return prepare_fixed_dataset(raw_root, split_csv, destination, pointer, recipe_path,
                                 new_source_split=new_source_split, dry_run=dry_run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR, help='Raw recording folders, not a previously generated dataset')
    parser.add_argument('--split-csv', type=Path, default=DEFAULT_SPLIT_CSV)
    output_options = parser.add_mutually_exclusive_group()
    output_options.add_argument('--output-dir', type=Path, help='Fixed generated folder; default: Dataset/Processed_5s; replaced on each successful run')
    output_options.add_argument('--versions-dir', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--pointer', type=Path, help='Default: Preprocessing/current_dataset.json')
    parser.add_argument('--config', type=Path, default=DEFAULT_RECIPE, help='Synthetic four-class generation recipe')
    parser.add_argument('--new-source-split', choices=SPLITS, default='train')
    parser.add_argument('--dry-run', action='store_true', help='Show discovery and assignments without writing any files')
    parser.add_argument('--force', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--verbose', action='store_true', help='Show recording progress and the full preparation summary')
    args = parser.parse_args(argv)
    progress = io.StringIO()
    try:
        with redirect_stdout(sys.stdout if args.verbose else progress):
            result = prepare_dataset(args.data_dir, args.split_csv, args.versions_dir, args.pointer,
                                     args.config, args.new_source_split, args.dry_run, args.force,
                                     output_dir=args.output_dir)
    except Exception:
        if not args.verbose:
            print(progress.getvalue(), end='', file=sys.stderr)
        raise
    if not args.verbose and not args.dry_run:
        print(f'CSV: {Path(result["data_dir"])/"samples.csv"}')
        return 0
    print('-'*64)
    print('Label mode: synthetic four-class; original recordings are assumed good.')
    display = {key: result[key] for key in ('status', 'recording_count', 'new_recordings',
                     'base_split_counts', 'split_counts', 'short_tail_count', 'manual_override_count',
                     'data_dir', 'version_dir')}
    if 'would_rebuild' in result:
        display['would_rebuild'] = result['would_rebuild']
    print(json.dumps(display, ensure_ascii=False, indent=2))
    print('Training is a separate manual step. Fresh main.py classification runs use the active prepared dataset.')
    print('-'*64)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
