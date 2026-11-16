"""Inspect raw recording folders and plan source-group assignments without writes."""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from Dataset.Preprocessing.build_segments import _original_source
from Dataset.Preprocessing.dataset_splits import CSV_FIELDS, SPLIT_NAMES, read_split_csv


DEFAULT_DATA_DIR = PROJECT_ROOT/'Dataset'/'Data'
DEFAULT_SPLIT_CSV = Path(__file__).resolve().parent/'dataset_splits.csv'
ALL_SPLITS = (*SPLIT_NAMES, 'exclude')
DERIVED_DIRECTORY_NAMES = {'segments_5s', 'fourclass_5s', 'CSV_exports', 'prepared_versions', 'window_demo'}
DERIVED_MARKERS = ('dataset_summary.json', 'segments.json', 'samples.json', 'demo_summary.json')
EXPECTED_WAV = 'audio_000000.wav'


def _is_derived(directory):
    return directory.name in DERIVED_DIRECTORY_NAMES or any((directory/name).is_file() for name in DERIVED_MARKERS)


def _file_stats(path):
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f'Expected a metadata file, found a directory: {path}')
    stat = path.stat()
    return {'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def _nested_wav(directory, has_direct_recording):
    """Find an unsupported nested recording, excluding known generated demos."""
    pending = [directory]
    while pending:
        current = pending.pop()
        for child in sorted(current.iterdir(), key=lambda path: path.name):
            if child.is_dir():
                if _is_derived(child) or (has_direct_recording and current == directory and child.name == 'clips'):
                    continue
                pending.append(child)
            elif current != directory and child.is_file() and child.suffix.lower() == '.wav':
                return child
    return None


def scan_recordings(data_dir=DEFAULT_DATA_DIR):
    """Return sorted raw-recording records, reading metadata and WAV headers only.

    Supported layout: one immediate subdirectory per recording, containing
    exactly audio_000000.wav and metadata.json. Generated outputs are ignored.
    A batch of nested recordings or other/multiple WAV names must be organized
    into the supported layout rather than silently omitted.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f'Data directory not found: {data_dir}')
    records = []
    for folder in sorted(data_dir.iterdir(), key=lambda path: path.name):
        if not folder.is_dir():
            if folder.is_file() and folder.suffix.lower() == '.wav':
                raise ValueError(f'Unsupported root-level WAV: {folder}. Put each recording in its own direct subdirectory.')
            continue
        if _is_derived(folder):
            continue
        direct_wavs = sorted((path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == '.wav'),
                             key=lambda path: path.name)
        if direct_wavs and (len(direct_wavs) != 1 or direct_wavs[0].name != EXPECTED_WAV):
            raise ValueError(f'Unsupported WAV layout in {folder}: expected one direct {EXPECTED_WAV}, found {[path.name for path in direct_wavs]}. Organize the recording before preprocessing.')
        nested = _nested_wav(folder, has_direct_recording=bool(direct_wavs))
        if nested is not None:
            raise ValueError(f'Unsupported nested WAV recording: {nested}. Move recording folders directly under {data_dir}.')
        if not direct_wavs:
            continue
        metadata_path = folder/'metadata.json'
        if not metadata_path.is_file():
            raise FileNotFoundError(f'Recording metadata not found: {metadata_path}')
        try:
            metadata = json.loads(metadata_path.read_text(encoding='utf-8-sig'))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f'Invalid metadata JSON: {metadata_path}: {error}') from error
        if not isinstance(metadata, dict):
            raise ValueError(f'Metadata must be a JSON object: {metadata_path}')
        source_id, original_start = _original_source(folder.name, metadata)
        wav_path = direct_wavs[0]
        try:
            with wave.open(str(wav_path), 'rb') as wav:
                sample_rate, channels, width, frames = wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()
                if (sample_rate, channels, width, wav.getcomptype()) != (44100, 2, 2, 'NONE'):
                    raise ValueError(f'Unsupported WAV format: {wav_path}; expected 44100 Hz stereo PCM16')
                if frames < 1:
                    raise ValueError(f'Empty WAV recording: {wav_path}; finish copying the recording before preprocessing')
        except (wave.Error, EOFError) as error:
            raise ValueError(f'Invalid WAV header: {wav_path}: {error}') from error
        signature = {
            'wav': {**_file_stats(wav_path), 'frame_count': frames},
            'metadata': _file_stats(metadata_path),
            'events': _file_stats(folder/'events.json'),
            'playback': _file_stats(folder/'playback.json'),
        }
        records.append({
            'recording_id': folder.name, 'source_recording_id': source_id,
            'directory': str(folder), 'wav_path': str(wav_path), 'metadata_path': str(metadata_path),
            'sample_rate_hz': sample_rate, 'channels': channels, 'bit_depth': width*8,
            'frame_count': frames, 'duration_seconds': frames/sample_rate,
            'original_recording_start_seconds': original_start, 'input_signature': signature,
        })
    return records


def plan_assignments(records, existing_rows, new_source_split='train'):
    """Keep existing rows and append assignments for new complete source groups.

    A new clip from a known active source inherits that split. An entirely
    excluded source stays excluded. Only an unseen source uses new_source_split.
    This function returns a plan and never updates the editable master CSV.
    """
    if new_source_split not in ALL_SPLITS:
        raise ValueError(f'new_source_split must be one of {ALL_SPLITS}')
    inventory = {}
    for record in records:
        identifier = record['recording_id']
        if identifier in inventory:
            raise ValueError(f'Duplicate scanned recording_id: {identifier}')
        inventory[identifier] = record
    assignments, seen = [], set()
    known_source_splits = {}
    all_old_sources = set()
    for original in existing_rows:
        row = {key: str(original.get(key, '')).strip() if original.get(key) is not None else '' for key in CSV_FIELDS}
        if not all(row.values()) or row['split'] not in ALL_SPLITS:
            raise ValueError('Existing split rows need recording_id, source_recording_id and a valid split')
        identifier, source, split = row['recording_id'], row['source_recording_id'], row['split']
        if identifier in seen:
            raise ValueError(f'Duplicate existing recording_id: {identifier}')
        seen.add(identifier)
        if identifier not in inventory:
            raise ValueError(f'Existing recording is missing from the raw inventory: {identifier}')
        actual_source = inventory[identifier]['source_recording_id']
        if actual_source != source:
            raise ValueError(f'Source metadata mismatch for {identifier}: CSV={source!r}, metadata={actual_source!r}')
        all_old_sources.add(source)
        if split in SPLIT_NAMES:
            previous = known_source_splits.get(source)
            if previous is not None and previous != 'exclude' and previous != split:
                raise ValueError(f'Source recording {source!r} appears across {previous} and {split}')
            known_source_splits[source] = split
        else:
            known_source_splits.setdefault(source, 'exclude')
        assignments.append(dict(row))
    new_recordings, added, new_sources = [], [], set()
    for identifier in sorted(set(inventory)-seen):
        source = inventory[identifier]['source_recording_id']
        split = known_source_splits.get(source, new_source_split)
        row = {'recording_id': identifier, 'source_recording_id': source, 'split': split}
        assignments.append(row)
        new_recordings.append(identifier)
        if source not in all_old_sources:
            new_sources.add(source)
            reason = 'new_source_group_default'
        elif split == 'exclude':
            reason = 'inherited_excluded_source'
        else:
            reason = 'inherited_active_source'
        added.append({**row, 'assignment_reason': reason})
        known_source_splits.setdefault(source, split)
    groups = {split: sorted({row['source_recording_id'] for row in assignments if row['split'] == split})
              for split in ALL_SPLITS}
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if set(groups[left]) & set(groups[right]):
            raise ValueError(f'Planned source groups cross {left}/{right}')
    return {
        'assignments': assignments, 'new_recordings': new_recordings,
        'new_source_recording_ids': sorted(new_sources), 'added_assignments': added,
        'source_groups': groups,
        'recording_count': len(records), 'existing_recording_count': len(existing_rows),
        'assignment_counts': {split: sum(row['split'] == split for row in assignments) for split in ALL_SPLITS},
        'recording_signatures': {identifier: inventory[identifier]['input_signature'] for identifier in sorted(inventory)},
        'active_source_groups_disjoint': True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--split-csv', type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument('--new-source-split', choices=ALL_SPLITS, default='train')
    args = parser.parse_args()
    records = scan_recordings(args.data_dir)
    plan = plan_assignments(records, read_split_csv(args.split_csv), args.new_source_split)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
