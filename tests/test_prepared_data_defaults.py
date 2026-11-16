"""Check dataset selection without training or reading production waveforms."""
import json
from pathlib import Path
import sys

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from Dataset.Preprocessing import active_dataset
from fourclass_training import _parser, _settings
from Validation.verify_classification import resolve_evaluation_data_dir


@pytest.fixture
def active_pointer(tmp_path, monkeypatch):
    pointer = tmp_path/'current_dataset.json'
    monkeypatch.setattr(active_dataset, 'ACTIVE_DATASET_POINTER', pointer)
    return pointer


def point_to(pointer, data_dir):
    pointer.write_text(json.dumps({'schema_version': 1, 'data_dir': str(data_dir)}))


def test_fresh_training_resolves_active_pointer_on_every_call(tmp_path, active_pointer):
    first = tmp_path/'first prepared version'
    second = tmp_path/'second prepared version'
    args = _parser().parse_args([])
    point_to(active_pointer, first)
    settings = _settings(args, None)
    assert settings['data_dir'] == str(first.resolve())
    assert settings['csv'] is None
    point_to(active_pointer, second)
    assert _settings(args, None)['data_dir'] == str(second.resolve())


def test_explicit_training_directory_bypasses_bad_active_pointer(tmp_path, active_pointer):
    active_pointer.write_text('{invalid json')
    explicit = tmp_path/'explicit prepared data'
    args = _parser().parse_args(['--data-dir', str(explicit)])
    assert _settings(args, None)['data_dir'] == str(explicit.resolve())


def test_csv_only_training_infers_dataset_parent_without_reading_active_pointer(tmp_path, active_pointer):
    active_pointer.write_text('{invalid json')
    csv_path = tmp_path/'selected dataset'/'samples.csv'
    settings = _settings(_parser().parse_args(['--csv', str(csv_path)]), None)
    assert settings['data_dir'] == str(csv_path.parent.resolve())
    assert settings['csv'] == str(csv_path.resolve())


def test_explicit_training_directory_wins_over_combined_csv_parent(tmp_path, active_pointer):
    active_pointer.write_text('{invalid json')
    data_dir = tmp_path/'actual waveform data'
    csv_path = tmp_path/'edited table copies'/'samples.csv'
    args = _parser().parse_args(['--csv', str(csv_path), '--data-dir', str(data_dir)])
    settings = _settings(args, None)
    assert settings['data_dir'] == str(data_dir.resolve())
    assert settings['csv'] == str(csv_path.resolve())


def test_resume_retains_saved_combined_csv_and_rejects_another_path(tmp_path, active_pointer):
    active_pointer.write_text('{invalid json')
    saved_csv = tmp_path/'historical dataset'/'samples.csv'
    previous = {'settings': _settings(_parser().parse_args(['--csv', str(saved_csv)]), None)}
    restored = _settings(_parser().parse_args(['--epochs', '40']), previous)
    assert restored['data_dir'] == str(saved_csv.parent.resolve())
    assert restored['csv'] == str(saved_csv.resolve())
    assert restored['epochs'] == 40
    with pytest.raises(ValueError, match='Resume CSV differs'):
        _settings(_parser().parse_args(['--csv', str(tmp_path/'other.csv')]), previous)


def test_legacy_resume_without_csv_retains_three_file_default(tmp_path, active_pointer):
    active_pointer.write_text('{invalid json')
    historical = tmp_path/'historical dataset'
    settings = _settings(_parser().parse_args(['--data-dir', str(historical)]), None)
    settings.pop('csv', None)
    restored = _settings(_parser().parse_args([]), {'settings': settings})
    assert restored['data_dir'] == str(historical.resolve())
    assert restored['csv'] is None


def test_resume_retains_saved_dataset_when_active_pointer_changes(tmp_path, active_pointer):
    historical = tmp_path/'historical prepared data'
    previous = {'settings': _settings(_parser().parse_args(['--data-dir', str(historical)]), None)}
    args = _parser().parse_args(['--epochs', '40'])
    point_to(active_pointer, tmp_path/'new current data')
    settings = _settings(args, previous)
    assert settings['data_dir'] == str(historical.resolve())
    assert settings['epochs'] == 40
    active_pointer.write_text('{invalid json')
    assert _settings(args, previous)['data_dir'] == str(historical.resolve())


def test_evaluation_prefers_checkpoint_settings_over_current_dataset_and_identity(tmp_path, active_pointer):
    historical = tmp_path/'historical prepared data'
    point_to(active_pointer, tmp_path/'new active data')
    checkpoint = {'training_state': {
        'settings': {'data_dir': str(historical)},
        'data_identity': {'data_dir': str(tmp_path/'identity fallback')},
    }}
    assert resolve_evaluation_data_dir(checkpoint) == historical.resolve()


def test_evaluation_falls_back_to_checkpoint_data_identity(tmp_path, active_pointer):
    historical = tmp_path/'historical prepared data'
    point_to(active_pointer, tmp_path/'new active data')
    checkpoint = {'training_state': {'data_identity': {'data_dir': str(historical)}}}
    assert resolve_evaluation_data_dir(checkpoint) == historical.resolve()


def test_explicit_evaluation_directory_overrides_checkpoint_and_bad_pointer(tmp_path, active_pointer):
    active_pointer.write_text('{invalid json')
    explicit = tmp_path/'explicit evaluation data'
    checkpoint = {'training_state': {'settings': {'data_dir': str(tmp_path/'historical data')}}}
    assert resolve_evaluation_data_dir(checkpoint, explicit) == explicit.resolve()
    assert resolve_evaluation_data_dir({}, explicit) == explicit.resolve()


@pytest.mark.parametrize('checkpoint', [{}, {'training_state': None}, {'training_state': {'settings': {}}}])
def test_evaluation_without_saved_dataset_requires_explicit_directory(checkpoint, tmp_path, active_pointer):
    point_to(active_pointer, tmp_path/'valid but unrelated active data')
    with pytest.raises(ValueError, match='provide --data-dir explicitly'):
        resolve_evaluation_data_dir(checkpoint)
