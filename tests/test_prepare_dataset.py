"""Exercise versioned preparation with small real WAV fixtures, never user data."""
import csv
import importlib
import json
from pathlib import Path
import sys
import wave

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from Dataset.Preprocessing.active_dataset import resolve_dataset_dir
from Dataset.Preprocessing import active_dataset
from Dataset.Preprocessing.dataset_splits import read_split_csv, write_split_csv


def write_raw(data_dir, recording_id, *, source_id=None, amplitude=10, sample_rate=44100):
    folder = data_dir/recording_id
    folder.mkdir(parents=True, exist_ok=True)
    time = np.arange(5*sample_rate, dtype=np.float64)/sample_rate
    pcm = np.column_stack((np.rint(8*np.sin(2*np.pi*71*time)),
                           np.rint(amplitude*np.sin(2*np.pi*31*time)))).astype('<i2')
    with wave.open(str(folder/'audio_000000.wav'), 'wb') as wav:
        wav.setframerate(sample_rate); wav.setnchannels(2); wav.setsampwidth(2)
        wav.writeframes(pcm.tobytes())
    metadata = {'Context': {'Operation': 'small test fixture'}}
    if source_id is not None:
        metadata['SourceRecording'] = {'Folder': f'C:\\capture\\{source_id}',
                                       'StartSeconds': 5, 'EndSeconds': 10}
    (folder/'metadata.json').write_text(json.dumps(metadata))
    return folder


@pytest.fixture
def preparation_paths(tmp_path, monkeypatch):
    data = tmp_path/'Dataset'/'Data'
    preprocessing = tmp_path/'Dataset'/'Preprocessing'
    preprocessing.mkdir(parents=True)
    records = [('train source', 'train'), ('val source', 'val'), ('test source', 'test')]
    for name, _ in records:
        write_raw(data, name)
    splits = preprocessing/'dataset_splits.csv'
    write_split_csv(splits, [{'recording_id': name, 'source_recording_id': name, 'split': split}
                             for name, split in records])
    recipe = preprocessing/'fourclass_config.json'
    recipe.write_text((PROJECT/'Dataset/Preprocessing/fourclass_config.json').read_text())
    monkeypatch.setattr(active_dataset, 'ACTIVE_DATASET_POINTER', preprocessing/'current_dataset.json')
    return dict(data_dir=data, split_csv=splits, versions_dir=data/'prepared_versions',
                pointer_path=preprocessing/'current_dataset.json', recipe_path=recipe)


def prepare(paths, **options):
    pipeline = importlib.import_module('Dataset.Preprocessing.prepare_dataset')
    return pipeline.prepare_dataset(**paths, **options)


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def verify_version(summary, paths, counts):
    assert summary['status'] == 'prepared'
    active = Path(summary['data_dir']).resolve()
    assert active == resolve_dataset_dir(pointer_path=paths['pointer_path'])
    assert active == resolve_dataset_dir()
    assert active.name == 'fourclass_5s'
    assert active.parent == Path(summary['version_dir']).resolve()
    assert active.parent.parent == paths['versions_dir']
    assert (active.parent/'segments_5s'/'segments.json').is_file()
    samples = json.loads((active/'samples.json').read_text())
    assert len(samples) == sum(counts.values())
    for split, count in counts.items():
        assert len(read_rows(active/f'{split}.csv')) == count
    for sample in samples:
        with wave.open(str(active/sample['wav_file']), 'rb') as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) == (44100, 2, 2, 220500)
    return active


def test_new_source_build_updates_default_resolver_without_changing_old_assignments(preparation_paths):
    paths = preparation_paths
    first = verify_version(prepare(paths), paths, {'train': 4, 'val': 4, 'test': 4})
    old_splits = {row['recording_id']: row for row in read_split_csv(paths['split_csv'])}
    first_samples = (first/'samples.json').read_bytes()
    first_pointer = paths['pointer_path'].read_bytes()
    write_raw(paths['data_dir'], 'new source')
    second = verify_version(prepare(paths), paths, {'train': 8, 'val': 4, 'test': 4})
    current = {row['recording_id']: row for row in read_split_csv(paths['split_csv'])}
    assert all(current[key] == value for key, value in old_splits.items())
    assert current['new source'] == {'recording_id': 'new source', 'source_recording_id': 'new source', 'split': 'train'}
    assert second != first and paths['pointer_path'].read_bytes() != first_pointer
    assert (first/'samples.json').read_bytes() == first_samples
    assert len(list(first.rglob('*.wav'))) == 12
    # Explicit older directories stay usable; the implicit resolver changes.
    assert resolve_dataset_dir(first, pointer_path=paths['pointer_path']) == first


def test_new_clip_inherits_existing_source_split_instead_of_default_train(preparation_paths):
    paths = preparation_paths
    prepare(paths)
    write_raw(paths['data_dir'], 'another val clip', source_id='val source')
    verify_version(prepare(paths), paths, {'train': 4, 'val': 8, 'test': 4})
    row = next(row for row in read_split_csv(paths['split_csv']) if row['recording_id'] == 'another val clip')
    assert row['source_recording_id'] == 'val source' and row['split'] == 'val'


def test_unchanged_is_noop_and_generated_directories_are_not_raw_inputs(preparation_paths):
    paths = preparation_paths
    write_raw(paths['data_dir']/'segments_5s'/'train', 'nested not raw')
    first = verify_version(prepare(paths), paths, {'train': 4, 'val': 4, 'test': 4})
    pointer = paths['pointer_path'].read_bytes()
    splits = paths['split_csv'].read_bytes()
    versions = set(paths['versions_dir'].iterdir())
    result = prepare(paths)
    assert result['status'] == 'unchanged'
    assert Path(result['data_dir']).resolve() == first
    assert paths['pointer_path'].read_bytes() == pointer
    assert paths['split_csv'].read_bytes() == splits
    assert set(paths['versions_dir'].iterdir()) == versions
    assert len(read_split_csv(paths['split_csv'])) == 3


def test_dry_run_writes_nothing_even_when_new_raw_source_is_found(preparation_paths):
    paths = preparation_paths
    write_raw(paths['data_dir'], 'new source')
    before = {str(path.relative_to(paths['data_dir'].parent)): (path.stat().st_size, path.stat().st_mtime_ns)
              for path in paths['data_dir'].parent.rglob('*') if path.is_file()}
    result = prepare(paths, dry_run=True)
    after = {str(path.relative_to(paths['data_dir'].parent)): (path.stat().st_size, path.stat().st_mtime_ns)
             for path in paths['data_dir'].parent.rglob('*') if path.is_file()}
    assert result['status'] == 'dry_run'
    assert after == before
    assert not paths['pointer_path'].exists() and not paths['versions_dir'].exists()
    assert len(read_split_csv(paths['split_csv'])) == 3


@pytest.mark.parametrize('bad_input', ['wrong_format', 'synthesis_overflow'])
def test_bad_input_does_not_replace_published_pointer_or_master_csv(preparation_paths, bad_input):
    paths = preparation_paths
    active = verify_version(prepare(paths), paths, {'train': 4, 'val': 4, 'test': 4})
    pointer, splits = paths['pointer_path'].read_bytes(), paths['split_csv'].read_bytes()
    original = (active/'samples.json').read_bytes()
    if bad_input == 'wrong_format':
        write_raw(paths['data_dir'], 'invalid source', sample_rate=48000)
    else:
        # Valid PCM16 input, but the configured synthesis amplitude will overflow.
        write_raw(paths['data_dir'], 'invalid source', amplitude=30000)
    with pytest.raises((ValueError, RuntimeError)):
        prepare(paths)
    assert paths['pointer_path'].read_bytes() == pointer
    assert paths['split_csv'].read_bytes() == splits
    assert resolve_dataset_dir(pointer_path=paths['pointer_path']) == active
    assert (active/'samples.json').read_bytes() == original


def swap_first_training_targets(active):
    path = active/'train.csv'
    rows = read_rows(path)
    rows[0]['label'], rows[1]['label'] = rows[1]['label'], rows[0]['label']
    rows[0]['label_index'], rows[1]['label_index'] = rows[1]['label_index'], rows[0]['label_index']
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return {row['sample_id']: (row['label'], row['label_index']) for row in rows}


def test_manual_csv_targets_survive_new_source_build(preparation_paths):
    paths = preparation_paths
    first = verify_version(prepare(paths), paths, {'train': 4, 'val': 4, 'test': 4})
    overrides = swap_first_training_targets(first)
    old_csv = (first/'train.csv').read_bytes()
    write_raw(paths['data_dir'], 'new source')
    second = verify_version(prepare(paths), paths, {'train': 8, 'val': 4, 'test': 4})
    current = {row['sample_id']: (row['label'], row['label_index']) for row in read_rows(second/'train.csv')}
    assert all(current[key] == value for key, value in overrides.items())
    assert (first/'train.csv').read_bytes() == old_csv


def test_manual_target_is_not_silently_carried_to_changed_base_pcm(preparation_paths):
    paths = preparation_paths
    active = verify_version(prepare(paths), paths, {'train': 4, 'val': 4, 'test': 4})
    swap_first_training_targets(active)
    pointer, splits = paths['pointer_path'].read_bytes(), paths['split_csv'].read_bytes()
    write_raw(paths['data_dir'], 'train source', amplitude=11)
    with pytest.raises((ValueError, RuntimeError)):
        prepare(paths)
    assert paths['pointer_path'].read_bytes() == pointer
    assert paths['split_csv'].read_bytes() == splits


def test_noncanonical_active_class_names_are_not_reset(preparation_paths):
    paths = preparation_paths
    active = verify_version(prepare(paths), paths, {'train': 4, 'val': 4, 'test': 4})
    names = json.loads((active/'labels.json').read_text())
    names['0'] = 'manually renamed good'
    (active/'labels.json').write_text(json.dumps(names))
    pointer, splits = paths['pointer_path'].read_bytes(), paths['split_csv'].read_bytes()
    with pytest.raises((ValueError, RuntimeError)):
        prepare(paths)
    assert paths['pointer_path'].read_bytes() == pointer
    assert paths['split_csv'].read_bytes() == splits
    assert json.loads((active/'labels.json').read_text()) == names


def test_force_builds_a_new_version_without_overwriting_the_previous_one(preparation_paths):
    paths = preparation_paths
    first = verify_version(prepare(paths), paths, {'train': 4, 'val': 4, 'test': 4})
    second = verify_version(prepare(paths, force=True), paths, {'train': 4, 'val': 4, 'test': 4})
    assert second != first and first.is_dir()


def test_pointer_publication_failure_restores_master_csv_and_old_pointer(preparation_paths, monkeypatch):
    paths = preparation_paths
    active = verify_version(prepare(paths), paths, {'train': 4, 'val': 4, 'test': 4})
    pointer_before = paths['pointer_path'].read_bytes()
    splits_before = paths['split_csv'].read_bytes()
    samples_before = (active/'samples.json').read_bytes()
    write_raw(paths['data_dir'], 'new source')
    pipeline = importlib.import_module('Dataset.Preprocessing.prepare_dataset')

    def fail_pointer_publication(data_dir, pointer_path=None):
        # The failure happens after the new master CSV has been published.
        assert len(read_split_csv(paths['split_csv'])) == 4
        assert paths['split_csv'].read_bytes() != splits_before
        # Simulate a failure after touching the pointer as well; both controls
        # must return to their original bytes rather than a partial new state.
        Path(pointer_path).write_text(json.dumps({'schema_version': 1, 'data_dir': str(data_dir)}))
        raise RuntimeError('simulated final pointer publication failure')

    monkeypatch.setattr(pipeline, 'set_active_dataset', fail_pointer_publication)
    with pytest.raises(RuntimeError, match='simulated final pointer publication failure'):
        prepare(paths)
    assert paths['split_csv'].read_bytes() == splits_before
    assert paths['pointer_path'].read_bytes() == pointer_before
    assert resolve_dataset_dir() == active
    assert (active/'samples.json').read_bytes() == samples_before
