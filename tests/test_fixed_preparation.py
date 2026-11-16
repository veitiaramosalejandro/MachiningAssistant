"""Fixed-output preparation uses real tiny datasets and never user recordings."""
import csv
import json
from pathlib import Path
import sys
import wave

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from Dataset.Preprocessing import fixed_preparation as fixed
from Dataset.Preprocessing.active_dataset import resolve_dataset_dir
from Dataset.Preprocessing.dataset_splits import read_split_csv, write_split_csv


def json_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def csv_read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def files(path):
    return {file.relative_to(path).as_posix(): file.read_bytes()
            for file in path.rglob('*') if file.is_file()}


def write_raw(data, name, *, amplitude=10, tail=0, source=None):
    folder = data/name; folder.mkdir(parents=True, exist_ok=True)
    count = 220500+tail
    t = np.arange(count)/44100
    pcm = np.column_stack((np.rint(8*np.sin(2*np.pi*71*t)),
                           np.rint(amplitude*np.sin(2*np.pi*31*t)))).astype('<i2')
    with wave.open(str(folder/'audio_000000.wav'), 'wb') as wav:
        wav.setnchannels(2); wav.setsampwidth(2); wav.setframerate(44100); wav.writeframes(pcm.tobytes())
    metadata = {'DurationSeconds': count/44100, 'Context': {'Operation': 'fixture'}}
    if source is not None:
        metadata['SourceRecording'] = {'Folder': 'D:\\raw\\'+source, 'StartSeconds': 0,
                                       'AudioOffsetMs': 1400}
    (folder/'metadata.json').write_text(json.dumps(metadata))
    (folder/'playback.json').write_text(json.dumps({'AudioOffsetMs': 0}))
    (folder/'events.json').write_text(json.dumps({'time_reference': 'playback_seconds', 'annotations': [
        {'id': name, 'timestamp_seconds': 0, 'machine_state': 'machining', 'quality': 3,
         'event_changes': {'machine_state': True, 'quality': True}},
    ]}))


@pytest.fixture
def paths(tmp_path):
    dataset = tmp_path/'Dataset'; data = dataset/'Data'
    preprocessing = dataset/'Preprocessing'; preprocessing.mkdir(parents=True)
    for name, tail in [('train source', 17), ('val source', 0), ('test source', 31)]:
        write_raw(data, name, tail=tail)
    splits = preprocessing/'dataset_splits.csv'
    write_split_csv(splits, [{'recording_id': name, 'source_recording_id': name, 'split': split}
                             for name, split in [('train source', 'train'), ('val source', 'val'), ('test source', 'test')]])
    recipe = preprocessing/'fourclass_config.json'
    recipe.write_text((PROJECT/'Dataset/Preprocessing/fourclass_config.json').read_text())
    return {'data_dir': data, 'split_csv': splits, 'output_dir': dataset/'Processed_5s',
            'pointer_path': preprocessing/'current_dataset.json', 'recipe_path': recipe}


def verify_output(paths, report, counts):
    output = paths['output_dir']
    assert report['status'] == 'prepared' and report['output_mode'] == 'replace'
    assert report['version_dir'] is None
    assert Path(report['data_dir']) == output and Path(report['output_dir']) == output
    assert resolve_dataset_dir(pointer_path=paths['pointer_path']) == output
    assert not (output/'fourclass_5s').exists()
    samples = json_read(output/'samples.json')
    good = {row['base_sample_id']: row['wav_file'] for row in samples if row['label'] == 'good'}
    for row in samples:
        assert row['base_wav'] == good[row['base_sample_id']]
        assert (output/row['wav_file']).is_file() and (output/row['base_wav']).is_file()
        assert (output/row['source_wav']).resolve() == paths['data_dir']/row['recording_id']/'audio_000000.wav'
        assert row['original_quality_score'] == 3
    for row in json_read(output/'segments.json'):
        assert row['wav_file'] == good[row['base_sample_id']]
        assert (output/row['source_wav']).is_file()
    tails = json_read(output/'tails.json')
    for row in tails:
        assert row['is_full_segment'] is False
        assert (output/row['wav_file']).is_file() and (output/row['source_wav']).is_file()
        with wave.open(str(output/row['wav_file']), 'rb') as wav:
            assert wav.getnframes() == row['frame_count']
    assert len(tails) == report['short_tail_count']
    for split, count in counts.items():
        assert len(csv_read(output/f'{split}.csv')) == count
    assert len(csv_read(output/'samples.csv')) == sum(counts.values())
    resolved = json_read(output/'resolved_config.json')
    assert resolved['reference_source_segments'] == 'segments.json'
    assert resolved['reference_split_snapshot'] == 'recording_splits.csv'
    assert (output/resolved['reference_source_segments']).is_file()
    assert (output/resolved['reference_split_snapshot']).is_file()
    base = json_read(output/'base_dataset_summary.json')
    assert Path(base['output_dir']) == output
    assert Path(base['data_dir']) == paths['data_dir']
    assert Path(base['split_csv']) == output/'recording_splits.csv'
    assert base['source_frame_count'] == base['full_frame_count']+base['tail_frame_count']
    summary = json_read(output/'dataset_summary.json')
    assert Path(summary['source_dir']) == output and Path(summary['output_dir']) == output
    for path in output.rglob('*'):
        if path.is_file() and path.suffix in {'.json', '.csv', '.txt'}:
            assert '.prepare_dataset_' not in path.read_text(encoding='utf-8-sig')
    assert not list(output.parent.glob('.prepare_dataset_*'))
    return output


def test_first_build_allows_empty_finder_folder_and_rebuild_replaces_same_path(paths, monkeypatch):
    output = paths['output_dir']; output.mkdir()
    (output/'.DS_Store').write_bytes(b'finder metadata only')
    original_raw = files(paths['data_dir'])
    calls = []
    original_builder = fixed.build_dataset_segments

    def tracked(*args, **kwargs):
        calls.append(True)
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(fixed, 'build_dataset_segments', tracked)
    first = fixed.prepare_fixed_dataset(**paths)
    verify_output(paths, first, {'train': 4, 'val': 4, 'test': 4})
    (output/'stale_generated_note.txt').write_text('must disappear on the next complete build')
    second = fixed.prepare_fixed_dataset(**paths)
    verify_output(paths, second, {'train': 4, 'val': 4, 'test': 4})
    assert len(calls) == 2
    assert not (output/'stale_generated_note.txt').exists()
    assert files(paths['data_dir']) == original_raw
    assert not (paths['data_dir']/'prepared_versions').exists()


def test_added_raw_source_and_manual_targets_survive_a_fixed_rebuild(paths):
    first = fixed.prepare_fixed_dataset(**paths)
    output = verify_output(paths, first, {'train': 4, 'val': 4, 'test': 4})
    old_assignments = read_split_csv(paths['split_csv'])
    rows = csv_read(output/'train.csv')
    rows[0]['label'], rows[1]['label'] = rows[1]['label'], rows[0]['label']
    rows[0]['label_index'], rows[1]['label_index'] = rows[1]['label_index'], rows[0]['label_index']
    edited = {row['sample_id']: (row['label'], row['label_index']) for row in rows}
    with (output/'train.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_raw(paths['data_dir'], 'new source', amplitude=14, tail=11)
    original_raw = files(paths['data_dir'])
    result = fixed.prepare_fixed_dataset(**paths)
    verify_output(paths, result, {'train': 8, 'val': 4, 'test': 4})
    assert result['manual_override_count'] == 2
    assert result['new_recordings'] == ['new source']
    assignments = read_split_csv(paths['split_csv'])
    assert assignments[:len(old_assignments)] == old_assignments
    assert assignments[-1] == {'recording_id': 'new source', 'source_recording_id': 'new source', 'split': 'train'}
    current = {row['sample_id']: (row['label'], row['label_index']) for row in csv_read(output/'train.csv')}
    assert all(current[key] == value for key, value in edited.items())
    assert files(paths['data_dir']) == original_raw


def test_dry_run_writes_no_files_or_directories(paths):
    write_raw(paths['data_dir'], 'new source')
    dataset_root = paths['data_dir'].parent
    before = files(dataset_root)
    directories = {str(path) for path in dataset_root.rglob('*') if path.is_dir()}
    report = fixed.prepare_fixed_dataset(**paths, dry_run=True)
    assert report['status'] == 'dry_run' and report['would_rebuild'] is True
    assert report['output_mode'] == 'replace' and report['version_dir'] is None
    assert files(dataset_root) == before
    assert {str(path) for path in dataset_root.rglob('*') if path.is_dir()} == directories
    assert not paths['output_dir'].exists() and not paths['pointer_path'].exists()


def test_pointer_publication_failure_restores_output_and_both_controls(paths, monkeypatch):
    fixed.prepare_fixed_dataset(**paths)
    output = paths['output_dir']
    old_output = files(output)
    old_split, old_pointer = paths['split_csv'].read_bytes(), paths['pointer_path'].read_bytes()
    write_raw(paths['data_dir'], 'new source')
    expected_raw = files(paths['data_dir'])

    def fail_after_pointer_write(directory, pointer_path=None):
        assert Path(directory) == output
        assert len(read_split_csv(paths['split_csv'])) == 4
        Path(pointer_path).write_text(json.dumps({'schema_version': 1, 'data_dir': 'incomplete publication'}))
        raise RuntimeError('simulated publication failure')

    monkeypatch.setattr(fixed, 'set_active_dataset', fail_after_pointer_write)
    with pytest.raises(RuntimeError, match='publication failure'):
        fixed.prepare_fixed_dataset(**paths)
    assert files(output) == old_output
    assert paths['split_csv'].read_bytes() == old_split
    assert paths['pointer_path'].read_bytes() == old_pointer
    assert files(paths['data_dir']) == expected_raw
    assert not list(output.parent.glob('.prepare_dataset_*'))


def test_failed_build_keeps_existing_output_and_controls(paths):
    fixed.prepare_fixed_dataset(**paths)
    output = paths['output_dir']
    old_output = files(output)
    old_split, old_pointer = paths['split_csv'].read_bytes(), paths['pointer_path'].read_bytes()
    write_raw(paths['data_dir'], 'overflow source', amplitude=30000)
    with pytest.raises(ValueError):
        fixed.prepare_fixed_dataset(**paths)
    assert files(output) == old_output
    assert paths['split_csv'].read_bytes() == old_split
    assert paths['pointer_path'].read_bytes() == old_pointer
    assert not list(output.parent.glob('.prepare_dataset_*'))


@pytest.mark.parametrize('kind', ['raw', 'ancestor', 'descendant', 'unmarked'])
def test_output_safety_preserves_raw_and_unrelated_directories(paths, kind):
    raw = paths['data_dir']
    output = {'raw': raw, 'ancestor': raw.parent, 'descendant': raw/'generated',
              'unmarked': raw.parent/'unrelated'}[kind]
    if kind == 'unmarked':
        output.mkdir(); (output/'keep.txt').write_text('not a generated dataset')
    before = files(raw.parent)
    with pytest.raises((ValueError, FileExistsError)):
        fixed.prepare_fixed_dataset(**{**paths, 'output_dir': output})
    assert files(raw.parent) == before
