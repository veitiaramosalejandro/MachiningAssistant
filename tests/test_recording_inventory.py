"""Read-only raw inventory and source-preserving split planning tests."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import wave

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from Dataset.Preprocessing.recording_inventory import plan_assignments, scan_recordings


def recording(root, name, source=None, *, rate=44100, channels=2, width=2, frames=13):
    folder = root/name; folder.mkdir(parents=True)
    metadata = {'Context': {'Machine': 'fixture'}}
    if source is not None:
        metadata['SourceRecording'] = {'Folder': f'D:\\raw\\{source}\\', 'StartSeconds': 1.25}
    (folder/'metadata.json').write_text(json.dumps(metadata))
    with wave.open(str(folder/'audio_000000.wav'), 'wb') as wav:
        wav.setnchannels(channels); wav.setsampwidth(width); wav.setframerate(rate)
        wav.writeframes(bytes(frames*channels*width))
    return folder


def assigned(name, split, source=None):
    return {'recording_id': name, 'source_recording_id': source or name, 'split': split}


def old_inventory(root):
    rows = []
    for name, split in [('train_old', 'train'), ('val_old', 'val'), ('test_old', 'test')]:
        recording(root, name)
        rows.append(assigned(name, split))
    return rows


def bytes_snapshot(root):
    return {path.relative_to(root): path.read_bytes() for path in root.rglob('*') if path.is_file()}


def test_new_source_records_are_grouped_and_old_rows_are_preserved_without_writes(tmp_path):
    old = old_inventory(tmp_path)
    recording(tmp_path, 'new_b', 'new_source')
    recording(tmp_path, 'new_a', 'new_source')
    original_rows, before = deepcopy(old), bytes_snapshot(tmp_path)
    records = scan_recordings(tmp_path)
    plan = plan_assignments(records, old)
    assert plan['assignments'][:3] == original_rows
    assert plan['new_recordings'] == ['new_a', 'new_b']
    assert plan['new_source_recording_ids'] == ['new_source']
    assert plan['assignments'][3:] == [assigned('new_a', 'train', 'new_source'), assigned('new_b', 'train', 'new_source')]
    assert plan['assignment_counts'] == {'train': 3, 'val': 1, 'test': 1, 'exclude': 0}
    assert old == original_rows and bytes_snapshot(tmp_path) == before
    assert plan['active_source_groups_disjoint']
    assert set(plan['recording_signatures']) == {record['recording_id'] for record in records}


def test_known_heldout_and_excluded_sources_are_inherited(tmp_path):
    old = old_inventory(tmp_path)
    recording(tmp_path, 'excluded_only', 'excluded_source')
    old.append(assigned('excluded_only', 'exclude', 'excluded_source'))
    recording(tmp_path, 'excluded_part_of_val', 'val_old')
    old.append(assigned('excluded_part_of_val', 'exclude', 'val_old'))
    recording(tmp_path, 'new_from_val', 'val_old')
    recording(tmp_path, 'new_from_test', 'test_old')
    recording(tmp_path, 'new_from_excluded', 'excluded_source')
    plan = plan_assignments(scan_recordings(tmp_path), old, new_source_split='train')
    result = {row['recording_id']: row for row in plan['assignments']}
    assert result['new_from_val']['split'] == 'val'
    assert result['new_from_test']['split'] == 'test'
    assert result['new_from_excluded']['split'] == 'exclude'
    assert result['excluded_part_of_val']['split'] == 'exclude'
    assert plan['new_source_recording_ids'] == []


@pytest.mark.parametrize('new_split', ['train', 'val', 'test', 'exclude'])
def test_new_source_split_is_explicitly_configurable(tmp_path, new_split):
    old = old_inventory(tmp_path)
    recording(tmp_path, 'new_record')
    plan = plan_assignments(scan_recordings(tmp_path), old, new_source_split=new_split)
    assert plan['assignments'][-1] == assigned('new_record', new_split)


def test_derived_directories_and_recording_demos_are_not_new_raw_sources(tmp_path):
    raw = recording(tmp_path, 'real_record')
    for name in ['segments_5s', 'fourclass_5s', 'CSV_exports', 'prepared_versions', 'window_demo']:
        folder = tmp_path/name/'nested'; folder.mkdir(parents=True)
        (folder/'audio_000000.wav').write_bytes(b'derived data')
    for index, marker in enumerate(['dataset_summary.json', 'segments.json', 'samples.json']):
        folder = tmp_path/f'generated_{index}'; folder.mkdir()
        (folder/marker).write_text('{}')
        (folder/'audio_000000.wav').write_bytes(b'derived data')
    for nested in ['window_demo/clips', 'clips']:
        folder = raw/nested; folder.mkdir(parents=True)
        (folder/'clip.wav').write_bytes(b'derived data')
    (tmp_path/'ordinary_notes'/'subfolder').mkdir(parents=True)
    (tmp_path/'ordinary_notes'/'subfolder'/'notes.txt').write_text('No source WAV')
    found = scan_recordings(tmp_path)
    assert [record['recording_id'] for record in found] == ['real_record']
    assert found[0]['source_recording_id'] == 'real_record'
    assert found[0]['frame_count'] == 13


@pytest.mark.parametrize('layout', ['wrong_name', 'multiple', 'deep_batch', 'nested_raw_under_record', 'root_wav'])
def test_unsupported_wav_layouts_are_not_silently_ignored(tmp_path, layout):
    if layout == 'root_wav':
        (tmp_path/'loose.wav').write_bytes(b'WAV in wrong location')
    elif layout == 'deep_batch':
        recording(tmp_path/'batch', 'nested_record')
    elif layout == 'nested_raw_under_record':
        raw = recording(tmp_path, 'direct_record')
        recording(raw/'another_batch', 'nested_record')
    else:
        raw = recording(tmp_path, 'record')
        if layout == 'wrong_name':
            (raw/'audio_000000.wav').rename(raw/'different_name.wav')
        else:
            (raw/'audio_000001.wav').write_bytes((raw/'audio_000000.wav').read_bytes())
    before = bytes_snapshot(tmp_path)
    with pytest.raises(ValueError, match='Unsupported'):
        scan_recordings(tmp_path)
    assert bytes_snapshot(tmp_path) == before


@pytest.mark.parametrize('metadata', ['not JSON', '[]', '{"SourceRecording":{"Folder":""}}'])
def test_invalid_source_metadata_is_reported_without_changes(tmp_path, metadata):
    raw = recording(tmp_path, 'record')
    (raw/'metadata.json').write_text(metadata)
    before = bytes_snapshot(tmp_path)
    with pytest.raises(ValueError):
        scan_recordings(tmp_path)
    assert bytes_snapshot(tmp_path) == before


def test_missing_recording_metadata_is_reported(tmp_path):
    raw = recording(tmp_path, 'record')
    (raw/'metadata.json').unlink()
    with pytest.raises(FileNotFoundError, match='metadata'):
        scan_recordings(tmp_path)


@pytest.mark.parametrize('rate,channels,width', [(22050, 2, 2), (44100, 1, 2), (44100, 2, 1)])
def test_wrong_wav_format_is_rejected_without_conversion(tmp_path, rate, channels, width):
    recording(tmp_path, 'record', rate=rate, channels=channels, width=width)
    before = bytes_snapshot(tmp_path)
    with pytest.raises(ValueError, match='44100 Hz stereo PCM16'):
        scan_recordings(tmp_path)
    assert bytes_snapshot(tmp_path) == before


def test_missing_old_record_or_changed_source_mapping_is_an_error(tmp_path):
    old = old_inventory(tmp_path)
    records = scan_recordings(tmp_path)
    with pytest.raises(ValueError, match='missing from the raw inventory'):
        plan_assignments(records, old+[assigned('missing_record', 'train')])
    changed = deepcopy(old); changed[0]['source_recording_id'] = 'different_original'
    with pytest.raises(ValueError, match='Source metadata mismatch'):
        plan_assignments(records, changed)


def test_conflicting_existing_source_assignments_are_rejected(tmp_path):
    recording(tmp_path, 'train_clip', 'same_origin')
    recording(tmp_path, 'test_clip', 'same_origin')
    rows = [assigned('train_clip', 'train', 'same_origin'), assigned('test_clip', 'test', 'same_origin')]
    with pytest.raises(ValueError, match='appears across'):
        plan_assignments(scan_recordings(tmp_path), rows)


def test_input_signatures_track_only_relevant_source_files(tmp_path):
    raw = recording(tmp_path, 'record', source='source_record')
    first = scan_recordings(tmp_path)[0]
    assert first['original_recording_start_seconds'] == 1.25
    signature = first['input_signature']
    assert signature['wav']['frame_count'] == 13
    assert set(signature['wav']) == {'size_bytes', 'mtime_ns', 'frame_count'}
    assert signature['events'] is None and signature['playback'] is None
    assert scan_recordings(tmp_path)[0]['input_signature'] == signature
    demo = raw/'window_demo'/'clips'; demo.mkdir(parents=True)
    (demo/'clip.wav').write_bytes(b'generated, not a source input')
    assert scan_recordings(tmp_path)[0]['input_signature'] == signature
    (raw/'events.json').write_text('{"annotations":[]}')
    (raw/'playback.json').write_text('{"AudioOffsetMs":0}')
    changed = scan_recordings(tmp_path)[0]['input_signature']
    assert changed['events']['size_bytes'] > 0 and changed['playback']['mtime_ns'] > 0
    assert changed['wav'] == signature['wav'] and changed['metadata'] == signature['metadata']
    metadata = json.loads((raw/'metadata.json').read_text()); metadata['Notes'] = 'Changed annotation context'
    (raw/'metadata.json').write_text(json.dumps(metadata))
    assert scan_recordings(tmp_path)[0]['input_signature']['metadata'] != signature['metadata']
