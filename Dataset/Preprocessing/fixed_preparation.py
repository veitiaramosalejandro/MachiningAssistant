"""Build and publish one replaceable, self-contained generated dataset directory."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile

from .active_dataset import resolve_dataset_dir, set_active_dataset
from .build_fourclass import FIELDS as SAMPLE_FIELDS, build_fourclass_dataset
from .build_segments import CSV_FIELDS as BASE_FIELDS, build_dataset_segments
from .dataset_splits import read_split_csv, write_split_csv


SPLITS = ('train', 'val', 'test', 'exclude')


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def _write_rows(path, rows, fields):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False, separators=(',', ':'))
                             if isinstance(row.get(key), (list, dict)) else row.get(key)
                             for key in fields})


def _relative(path, directory):
    return Path(os.path.relpath(path, directory)).as_posix()


def _generated_markers(path):
    return path.is_dir() and (path/'labels.json').is_file() and (path/'samples.json').is_file()


def _check_output_location(raw, output):
    if output == raw or output in raw.parents or raw in output.parents:
        raise ValueError('Generated output must be separate from raw data: not equal, an ancestor, or a descendant')
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f'Generated output is not a directory: {output}')
        meaningful = [path for path in output.iterdir() if not (path.name == '.DS_Store' and path.is_file())]
        if meaningful and not _generated_markers(output):
            raise FileExistsError(f'Refusing to replace non-generated output without labels.json and samples.json: {output}')


def _relocate_build(base, staged, final, raw, records, snapshot, recipe):
    """Keep all base references inside the completed build and raw paths stable."""
    source_wavs = {row['recording_id']: Path(row['wav_path']) for row in records}
    samples = _json(staged/'samples.json')
    good = {row['base_sample_id']: row['wav_file'] for row in samples if row['label'] == 'good'}
    for row in samples:
        row['base_wav'] = good[row['base_sample_id']]
        row['source_wav'] = _relative(source_wavs[row['recording_id']], final)
    _write_json(staged/'samples.json', samples)
    _write_rows(staged/'samples.csv', samples, SAMPLE_FIELDS)
    for split in SPLITS:
        _write_rows(staged/('excluded.csv' if split == 'exclude' else f'{split}.csv'),
                    [row for row in samples if row['split'] == split], SAMPLE_FIELDS)

    segments = _json(base/'segments.json')
    tails = _json(base/'tails.json')
    for row in segments:
        row['wav_file'] = good[row['base_sample_id']]
        row['source_wav'] = _relative(source_wavs[row['recording_id']], final)
    for row in tails:
        row['source_wav'] = _relative(source_wavs[row['recording_id']], final)
    if (base/'tails').is_dir():
        shutil.copytree(base/'tails', staged/'tails')
    _write_json(staged/'segments.json', segments)
    _write_rows(staged/'segments.csv', segments, BASE_FIELDS)
    _write_json(staged/'tails.json', tails)
    _write_rows(staged/'tails.csv', tails, BASE_FIELDS)
    # Preserve the original base-builder snapshot name as well as the fixed
    # pipeline's explicit recording snapshot used by resolved_config.json.
    shutil.copyfile(snapshot, staged/'recording_splits.csv')
    shutil.copyfile(base/'dataset_splits_used.csv', staged/'dataset_splits_used.csv')
    _write_json(staged/'recipe_used.json', recipe)

    base_summary = _json(base/'dataset_summary.json')
    base_summary.update(data_dir=str(raw), output_dir=str(final), split_csv=str(final/'recording_splits.csv'))
    for record in base_summary['recordings']:
        record['source_wav'] = _relative(source_wavs[record['recording_id']], final)
    _write_json(staged/'base_dataset_summary.json', base_summary)
    summary = _json(staged/'dataset_summary.json')
    summary.update(source_dir=str(final), output_dir=str(final), raw_data_dir=str(raw),
                   output_mode='replace', base_metadata_file='segments.json',
                   base_summary_file='base_dataset_summary.json', tail_count=len(tails),
                   tails_file='tails.json', tails_preserved=True)
    _write_json(staged/'dataset_summary.json', summary)
    resolved = _json(staged/'resolved_config.json')
    resolved['reference_source_segments'] = 'segments.json'
    resolved['reference_split_snapshot'] = 'recording_splits.csv'
    _write_json(staged/'resolved_config.json', resolved)
    return samples, segments, tails


def _validate_paths(staged, final, records, samples, segments, tails, workspace):
    raw_wavs = {row['recording_id']: Path(row['wav_path']).resolve() for row in records}
    for row in samples:
        for field in ('wav_file', 'base_wav'):
            relative = Path(row[field])
            path = (staged/relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(staged) or not path.is_file():
                raise ValueError(f'Generated {field} is not self-contained: {row[field]}')
    for row in [*samples, *segments, *tails]:
        relative = Path(row['wav_file'])
        path = (staged/relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(staged) or not path.is_file():
            raise ValueError(f'Generated WAV reference cannot be resolved: {row["wav_file"]}')
        if (final/row['source_wav']).resolve() != raw_wavs[row['recording_id']]:
            raise ValueError(f'Raw source reference is not correctly rebased: {row["source_wav"]}')
    resolved = _json(staged/'resolved_config.json')
    for field in ('reference_source_segments', 'reference_split_snapshot'):
        if not (staged/resolved[field]).is_file():
            raise ValueError(f'Missing saved reference: {field}')
    for path in staged.rglob('*'):
        if path.is_file() and path.suffix.lower() in ('.json', '.csv', '.txt'):
            if str(workspace) in path.read_text(encoding='utf-8-sig'):
                raise ValueError(f'Temporary path remains in prepared artifact: {path.name}')


def _restore_control(path, original, pipeline):
    current = pipeline._bytes_if_present(path)
    if current == original:
        return
    if original is None:
        if path.exists():
            path.unlink()
    else:
        pipeline._write_bytes_atomic(path, original)


def prepare_fixed_dataset(data_dir, split_csv, output_dir, pointer_path, recipe_path,
                          new_source_split='train', dry_run=False):
    # Import inside the entry point: prepare_dataset's public dispatcher may
    # itself import this module, while its existing helpers remain reusable.
    from . import prepare_dataset as pipeline

    raw = Path(data_dir).expanduser().resolve()
    split_csv = Path(split_csv).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    pointer = Path(pointer_path).expanduser().resolve()
    recipe_path = Path(recipe_path).expanduser().resolve()
    _check_output_location(raw, output)
    split_before = split_csv.read_bytes()
    pointer_before = pipeline._bytes_if_present(pointer)
    recipe_before = recipe_path.read_bytes()
    recipe = _json(recipe_path)
    if pipeline._validated_config(recipe)['segment_frames'] != 220500:
        raise ValueError('This training workflow requires complete 5-second samples (220500 frames)')
    records = pipeline.scan_recordings(raw)
    plan = pipeline.plan_assignments(records, read_split_csv(split_csv), new_source_split)
    candidate = output if _generated_markers(output) else (
        resolve_dataset_dir(pointer_path=pointer) if pointer.exists() else output)
    active = candidate if _generated_markers(candidate) else None
    overrides = pipeline._manual_overrides(active) if active is not None else {}
    signatures = [{'recording_id': row['recording_id'], 'source_recording_id': row['source_recording_id'],
                   'input_signature': row['input_signature']} for row in records]
    override_signature = [{'key': list(key), 'label': value['label'], 'label_index': value['label_index']}
                          for key, value in sorted(overrides.items())]
    frame_counts = {row['recording_id']: row['frame_count'] for row in records}
    base_counts = {split: sum(frame_counts[row['recording_id']]//220500 for row in plan['assignments'] if row['split'] == split)
                   for split in SPLITS}
    report = {'schema_version': 1, 'status': 'dry_run' if dry_run else 'building',
              'output_mode': 'replace', 'version_dir': None, 'data_dir': str(output), 'output_dir': str(output),
              'raw_data_dir': str(raw), 'label_mode': 'synthetic_fourclass',
              'recording_count': len(records), 'new_recordings': plan['new_recordings'],
              'assignments': plan['assignments'], 'source_groups': plan['source_groups'],
              'new_source_split': new_source_split, 'base_split_counts': base_counts,
              'split_counts': {split: value*4 for split, value in base_counts.items()},
              'short_tail_count': sum(row['frame_count'] % 220500 != 0 for row in records),
              'manual_override_count': len(overrides), 'previous_active_data_dir': str(active) if active else None,
              'active_pointer': str(pointer), 'would_rebuild': True,
              'label_assumption': 'Original recordings are good by the selected synthetic-demo convention, not inferred from event quality scores.',
              'input_signature': {'recordings': signatures, 'assignments': plan['assignments'],
                                  'recipe': recipe, 'manual_overrides': override_signature}}
    if dry_run:
        return report

    output.parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix='.prepare_dataset_', dir=output.parent))
    base, staged, backup = workspace/'base', workspace/'output', workspace/'previous_output'
    try:
        snapshot = workspace/'recording_splits.csv'
        write_split_csv(snapshot, plan['assignments'])
        build_dataset_segments(raw, snapshot, base, segment_seconds=5.)
        build_fourclass_dataset(base, staged, recipe)
        samples, segments, tails = _relocate_build(base, staged, output, raw, records, snapshot, recipe)
        applied = pipeline._preserve_overrides(staged, overrides)
        from .fourclass_data import load_manifest
        manifest = load_manifest(staged)
        if len(manifest) != sum(report['split_counts'][split] for split in ('train', 'val', 'test')):
            raise ValueError('Prepared sample counts differ from the recording plan')
        after = pipeline.scan_recordings(raw)
        if [{'recording_id': row['recording_id'], 'source_recording_id': row['source_recording_id'],
             'input_signature': row['input_signature']} for row in after] != signatures:
            raise RuntimeError('Raw recordings changed during preprocessing; rerun after copying is finished')
        current_overrides = pipeline._manual_overrides(active) if active is not None else {}
        if [{'key': list(key), 'label': value['label'], 'label_index': value['label_index']}
            for key, value in sorted(current_overrides.items())] != override_signature:
            raise RuntimeError('Current CSV targets changed during preprocessing; rerun with the latest labels')
        if (split_csv.read_bytes() != split_before or pipeline._bytes_if_present(pointer) != pointer_before
                or recipe_path.read_bytes() != recipe_before):
            raise RuntimeError('Split table, recipe or active pointer changed during preprocessing')
        report.update(status='prepared', created_at_utc=datetime.now(timezone.utc).isoformat(),
                      manual_override_count=applied, reference_rms_pcm=_json(staged/'resolved_config.json')['reference_rms_pcm'])
        _write_json(staged/'preparation.json', report)
        (staged/'README.txt').write_text(f'''Fixed five-second prepared dataset
==================================

Generated samples: {len(samples)}. Original base segments: {len(segments)}.
Short tails retained separately: {len(tails)}. Manual CSV targets preserved: {applied}.
WAV format: 44100 Hz, stereo PCM16; each complete sample has 220500 frames.

samples.csv is the combined training index; train.csv, val.csv and test.csv are
companion indexes. labels.json defines the four synthetic-demo target names.
Original good waveforms retain their PCM bytes; other tiers add the configured
right-channel sine. These labels are not measured machining-quality grades.

segments.json/csv describe original bases and point to the corresponding good WAV
inside this same directory. Every generated sample's base_wav uses that good path.
tails/ and tails.json/csv retain incomplete originals without padding or synthesis.
base_dataset_summary.json records the full original-frame coverage.
source_wav paths resolve against this directory to the unchanged raw recordings.
recording_splits.csv and recipe_used.json record this build's input choices.
resolved_config.json refers to segments.json and recording_splits.csv locally.

Run Dataset/Preprocessing/prepare_dataset.py to rebuild this fixed directory.
A complete new build is validated before replacing it; raw recordings, model
checkpoints and other output directories are not rewritten by this operation.
Current CSV target edits within the same four classes are retained when their
original base PCM is unchanged; samples.json retains generation-time labels.
''', encoding='utf-8')
        _validate_paths(staged, output, records, samples, segments, tails, workspace)

        had_output = output.exists()
        installed = False
        try:
            if had_output:
                output.rename(backup)
            staged.rename(output)
            installed = True
            pipeline._write_bytes_atomic(split_csv, snapshot.read_bytes())
            set_active_dataset(output, pointer_path=pointer)
        except BaseException:
            # Only the newly installed generated folder is removed. An old
            # output remains in backup until its rename back has succeeded.
            try:
                if backup.exists():
                    if output.exists():
                        shutil.rmtree(output)
                    backup.rename(output)
                elif not had_output and output.exists() and (installed or not staged.exists()):
                    shutil.rmtree(output)
            finally:
                try:
                    _restore_control(split_csv, split_before, pipeline)
                finally:
                    _restore_control(pointer, pointer_before, pipeline)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return report
    finally:
        # If restoring an old output itself failed, retain its backup for
        # recovery rather than deleting the only surviving copy.
        if workspace.exists() and not backup.exists():
            shutil.rmtree(workspace)
