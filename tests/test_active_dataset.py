"""Active-version resolution and atomic publication use temporary datasets only."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import struct
import subprocess
import sys
import wave

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from Dataset.Preprocessing import active_dataset as active


@pytest.fixture
def isolated_project(tmp_path, monkeypatch):
    root = tmp_path/'project'
    root.mkdir()
    legacy = root/'Dataset/Data/fourclass_5s'
    pointer = root/'Dataset/Preprocessing/current_dataset.json'
    monkeypatch.setattr(active, 'PROJECT_ROOT', root)
    monkeypatch.setattr(active, 'LEGACY_DATA_DIR', legacy)
    monkeypatch.setattr(active, 'ACTIVE_DATASET_POINTER', pointer)
    return root, legacy, pointer


def make_dataset(path, names=('ready', 'working'), tag='sample'):
    path.mkdir(parents=True)
    (path/'labels.json').write_text(json.dumps({str(index): name for index, name in enumerate(names)}))
    rows = []
    for split, indices in [('train', range(len(names))), ('val', [0]), ('test', [len(names)-1])]:
        selected = []
        for index in indices:
            identifier = f'{tag}_{split}_{index}'
            relative = f'{split}/{identifier}.wav'
            wav_path = path/relative; wav_path.parent.mkdir(exist_ok=True)
            with wave.open(str(wav_path), 'wb') as wav:
                wav.setnchannels(2); wav.setsampwidth(2); wav.setframerate(44100)
                wav.writeframes(struct.pack('<16h', *([index, index+3]*8)))
            row = {'sample_id': identifier, 'base_sample_id': 'base_'+identifier,
                   'recording_id': f'{tag}_{split}_recording', 'source_recording_id': f'{tag}_{split}_source',
                   'split': split, 'label': names[index], 'label_index': index, 'wav_file': relative,
                   'frame_count': 8, 'start_frame': 0, 'end_frame_exclusive': 8,
                   'start_seconds': 0., 'end_seconds_exclusive': 8/44100, 'duration_seconds': 8/44100}
            selected.append(row); rows.append(row)
        with (path/f'{split}.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
            writer.writeheader(); writer.writerows(selected)
    (path/'samples.json').write_text(json.dumps(rows))
    return rows


def test_missing_pointer_falls_back_without_creating_files(isolated_project):
    _, legacy, pointer = isolated_project
    assert active.resolve_dataset_dir() == legacy.resolve()
    assert not pointer.exists() and not legacy.exists()


def test_explicit_path_does_not_read_a_broken_pointer(isolated_project, monkeypatch, tmp_path):
    _, _, pointer = isolated_project
    pointer.parent.mkdir(parents=True)
    pointer.write_text('not JSON')
    explicit = tmp_path/'explicit data'; explicit.mkdir()
    monkeypatch.chdir(tmp_path)
    assert active.resolve_dataset_dir('explicit data') == explicit.resolve()
    assert active.resolve_dataset_dir(explicit, pointer_path=pointer) == explicit.resolve()
    with pytest.raises(ValueError):
        active.resolve_dataset_dir()


@pytest.mark.parametrize('external', [False, True])
def test_publish_and_resolve_project_relative_or_absolute_targets(isolated_project, tmp_path, monkeypatch, external):
    root, _, pointer = isolated_project
    target = tmp_path/'external version' if external else root/'Dataset/Data/prepared_versions/version1/fourclass_5s'
    make_dataset(target)
    before = {file.relative_to(target): file.read_bytes() for file in target.rglob('*') if file.is_file()}
    assert active.set_active_dataset(target) == target.resolve()
    value = json.loads(pointer.read_text())
    assert value['schema_version'] == 1
    assert Path(value['data_dir']).is_absolute() is external
    if not external:
        assert value['data_dir'] == 'Dataset/Data/prepared_versions/version1/fourclass_5s'
    monkeypatch.chdir(tmp_path)
    assert active.resolve_dataset_dir() == target.resolve()
    assert before == {file.relative_to(target): file.read_bytes() for file in target.rglob('*') if file.is_file()}
    assert not pointer.with_name(pointer.name+'.tmp').exists()


def test_custom_pointer_relative_target_is_project_relative_not_pointer_relative(isolated_project, tmp_path):
    root, _, _ = isolated_project
    target = root/'Dataset/Data/prepared_versions/v2/fourclass_5s'; make_dataset(target)
    custom = tmp_path/'unrelated config'/'pointer.json'; custom.parent.mkdir()
    custom.write_text(json.dumps({'schema_version': 1, 'data_dir': str(target.relative_to(root))}))
    assert active.resolve_dataset_dir(pointer_path=custom) == target.resolve()


@pytest.mark.parametrize('problem', ['missing_labels', 'missing_csv', 'bad_header', 'bad_labels'])
def test_invalid_dataset_cannot_replace_a_working_pointer(isolated_project, problem):
    root, _, pointer = isolated_project
    first, invalid = root/'prepared1', root/'prepared2'
    make_dataset(first); make_dataset(invalid)
    active.set_active_dataset(first)
    before = pointer.read_bytes()
    if problem == 'missing_labels':
        (invalid/'labels.json').unlink()
    elif problem == 'missing_csv':
        (invalid/'val.csv').unlink()
    elif problem == 'bad_header':
        (invalid/'train.csv').write_text('sample_id,label\na,ready\n')
    else:
        (invalid/'labels.json').write_text(json.dumps({'0': 'same', '1': 'same'}))
    with pytest.raises((FileNotFoundError, ValueError)):
        active.set_active_dataset(invalid)
    assert pointer.read_bytes() == before
    assert active.resolve_dataset_dir() == first.resolve()


def test_failed_atomic_publish_preserves_the_previous_pointer(isolated_project, monkeypatch):
    root, _, pointer = isolated_project
    first, second = root/'version1', root/'version2'
    make_dataset(first); make_dataset(second)
    active.set_active_dataset(first)
    previous = pointer.read_bytes()
    original_replace = Path.replace

    def interrupted_replace(self, destination):
        if self == pointer.with_name(pointer.name+'.tmp'):
            raise OSError('simulated pointer publication failure')
        return original_replace(self, destination)

    monkeypatch.setattr(Path, 'replace', interrupted_replace)
    with pytest.raises(OSError, match='publication failure'):
        active.set_active_dataset(second)
    assert pointer.read_bytes() == previous
    assert active.resolve_dataset_dir() == first.resolve()
    assert not pointer.with_name(pointer.name+'.tmp').exists()


@pytest.mark.parametrize('value', [[], {'schema_version': 2, 'data_dir': 'somewhere'},
                                   {'schema_version': 1, 'data_dir': ''}])
def test_invalid_existing_pointer_is_not_silently_replaced_by_legacy(isolated_project, value):
    _, _, pointer = isolated_project
    pointer.parent.mkdir(parents=True)
    pointer.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        active.resolve_dataset_dir()


def test_loaders_follow_each_new_pointer_but_explicit_directory_wins(isolated_project):
    from Dataset.Preprocessing import fourclass_data
    root, _, _ = isolated_project
    old_default = fourclass_data.DEFAULT_DATA_DIR
    first, second = root/'v1', root/'v2'
    make_dataset(first, names=('a', 'b'), tag='one')
    make_dataset(second, names=('x', 'y', 'z'), tag='two')
    active.set_active_dataset(first)
    assert fourclass_data.load_class_names() == ('a', 'b')
    assert all(row['sample_id'].startswith('one_') for row in fourclass_data.load_manifest(valid_frames=8))
    active.set_active_dataset(second)
    assert fourclass_data.load_class_names() == ('x', 'y', 'z')
    assert all(row['sample_id'].startswith('two_') for row in fourclass_data.load_manifest(valid_frames=8))
    assert fourclass_data.load_class_names(first) == ('a', 'b')
    assert all(row['sample_id'].startswith('one_') for row in fourclass_data.load_manifest(first, valid_frames=8))
    assert fourclass_data.DEFAULT_DATA_DIR == old_default


def test_export_cli_uses_active_default_and_explicit_override(isolated_project, tmp_path, monkeypatch):
    from Dataset.Preprocessing import export_csv
    root, _, pointer = isolated_project
    current, explicit = root/'current', root/'explicit'
    make_dataset(current); make_dataset(explicit)
    active.set_active_dataset(current)
    calls = []

    def fake_export(data_dir, output_dir):
        calls.append((Path(data_dir), Path(output_dir)))
        return {'output_dir': str(output_dir), 'sample_count': 4,
                'split_counts': {'train': 2, 'val': 1, 'test': 1, 'exclude': 0},
                'label_authority': 'current_split_csvs'}

    monkeypatch.setattr(export_csv, 'export_csv_indexes', fake_export)
    monkeypatch.setattr(sys, 'argv', ['export_csv.py', '--output-dir', str(tmp_path/'review1')])
    export_csv.main()
    pointer.write_text('deliberately broken pointer')
    monkeypatch.setattr(sys, 'argv', ['export_csv.py', '--data-dir', str(explicit),
                                    '--output-dir', str(tmp_path/'review2')])
    export_csv.main()
    assert [data for data, _ in calls] == [current.resolve(), explicit.resolve()]


def test_active_publication_and_exporter_import_need_only_standard_library(tmp_path):
    target = tmp_path/'data'; make_dataset(target)
    pointer = tmp_path/'pointer.json'
    preprocessing = PROJECT/'Dataset/Preprocessing'
    script = f'''
import builtins, sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {{'torch', 'numpy', 'pandas', 'scipy'}}:
        raise AssertionError('Unexpected non-standard dependency: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
sys.path.insert(0, {str(preprocessing)!r})
import active_dataset
import export_csv
assert active_dataset.set_active_dataset({str(target)!r}, pointer_path={str(pointer)!r}) == active_dataset.resolve_dataset_dir(pointer_path={str(pointer)!r})
print('stdlib-only publication/import passed')
'''
    result = subprocess.run([sys.executable, '-B', '-I', '-c', script], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert 'stdlib-only publication/import passed' in result.stdout
