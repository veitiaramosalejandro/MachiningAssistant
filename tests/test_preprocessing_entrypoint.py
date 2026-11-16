"""Exercise preprocessing routing with mocks; never build data or start training."""
import importlib
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


@pytest.fixture
def entrypoint(tmp_path, monkeypatch):
    entry = importlib.import_module('Dataset.Preprocessing.prepare_dataset')
    fixed = ModuleType('Dataset.Preprocessing.fixed_preparation')
    calls = []
    raw = tmp_path/'Dataset'/'Data'

    def report(output_dir, dry_run):
        return {'status': 'dry_run' if dry_run else 'prepared',
                'data_dir': str(output_dir), 'version_dir': None,
                'output_mode': 'replace', 'recording_count': 3, 'new_recordings': [],
                'base_split_counts': {'train': 1, 'val': 1, 'test': 1, 'exclude': 0},
                'split_counts': {'train': 4, 'val': 4, 'test': 4, 'exclude': 0},
                'short_tail_count': 0, 'manual_override_count': 0,
                'csv_path': str(Path(output_dir)/'samples.csv'), 'would_rebuild': True}

    def fake_fixed(data_dir, split_csv, output_dir, pointer_path, recipe_path,
                   new_source_split='train', dry_run=False):
        calls.append({'route': 'fixed', 'data_dir': Path(data_dir), 'split_csv': split_csv,
                      'output_dir': Path(output_dir), 'pointer_path': pointer_path,
                      'recipe_path': recipe_path, 'new_source_split': new_source_split,
                      'dry_run': dry_run})
        print('internal fixture build progress')
        return report(output_dir, dry_run)

    def fake_versioned(data_dir, split_csv, versions_dir, pointer_path, recipe_path,
                       new_source_split='train', dry_run=False, force=False):
        calls.append({'route': 'versioned', 'data_dir': Path(data_dir),
                      'versions_dir': Path(versions_dir), 'dry_run': dry_run, 'force': force})
        result = report(Path(versions_dir)/'fixture version'/'fourclass_5s', dry_run)
        result.update(version_dir=str(Path(versions_dir)/'fixture version'), output_mode='versioned')
        return result

    def unexpected_subprocess(*args, **kwargs):
        pytest.fail('Preprocessing entrypoint must not start training or another subprocess')

    monkeypatch.setattr(subprocess, 'run', unexpected_subprocess)
    fixed.prepare_fixed_dataset = fake_fixed
    monkeypatch.setitem(sys.modules, fixed.__name__, fixed)
    monkeypatch.setattr(entry, '_prepare_versioned_dataset', fake_versioned)
    monkeypatch.setattr(entry, 'DEFAULT_DATA_DIR', raw)
    return SimpleNamespace(module=entry, calls=calls, raw=raw, temporary_root=tmp_path)


def test_default_api_routes_to_sibling_fixed_directory(entrypoint):
    app = entrypoint
    result = app.module.prepare_dataset(data_dir=app.raw, versions_dir=None)
    assert len(app.calls) == 1
    assert app.calls[0]['route'] == 'fixed'
    assert app.calls[0]['data_dir'] == app.raw.resolve()
    assert app.calls[0]['output_dir'] == app.raw.parent/'Processed_5s'
    assert result['csv_path'] == str(app.raw.parent/'Processed_5s'/'samples.csv')
    assert not list(app.temporary_root.iterdir())


def test_cli_default_is_quiet_and_prints_one_combined_csv_path(entrypoint, capsys):
    app = entrypoint
    assert app.module.main([]) == 0
    assert len(app.calls) == 1 and app.calls[0]['route'] == 'fixed'
    assert capsys.readouterr().out == f'CSV: {app.raw.parent / "Processed_5s" / "samples.csv"}\n'
    assert not list(app.temporary_root.iterdir())


def test_cli_explicit_output_directory_is_forwarded_without_versioning(entrypoint, capsys):
    app = entrypoint
    output = app.temporary_root/'output with spaces'
    assert app.module.main(['--data-dir', str(app.raw), '--output-dir', str(output)]) == 0
    assert len(app.calls) == 1
    assert app.calls[0]['route'] == 'fixed'
    assert app.calls[0]['output_dir'] == output.resolve()
    assert capsys.readouterr().out == f'CSV: {output / "samples.csv"}\n'
    assert not list(app.temporary_root.iterdir())


def test_explicit_versions_directory_preserves_legacy_route(entrypoint):
    app = entrypoint
    versions = app.temporary_root/'preserved versions'
    result = app.module.prepare_dataset(data_dir=app.raw, versions_dir=versions, force=True)
    assert app.calls == [{'route': 'versioned', 'data_dir': app.raw.resolve(),
                          'versions_dir': versions.resolve(), 'dry_run': False, 'force': True}]
    assert result['version_dir'] == str(versions/'fixture version')
    assert not list(app.temporary_root.iterdir())


def test_fixed_and_versioned_output_arguments_cannot_be_combined(entrypoint):
    app = entrypoint
    with pytest.raises(ValueError, match='output|versions'):
        app.module.prepare_dataset(data_dir=app.raw, output_dir=app.temporary_root/'fixed',
                                   versions_dir=app.temporary_root/'versions')
    assert not app.calls
    assert not list(app.temporary_root.iterdir())


def test_cli_dry_run_only_forwards_a_plan_without_writing(entrypoint, capsys):
    app = entrypoint
    assert app.module.main(['--data-dir', str(app.raw), '--dry-run']) == 0
    assert len(app.calls) == 1 and app.calls[0]['route'] == 'fixed'
    assert app.calls[0]['dry_run'] is True
    assert 'dry_run' in capsys.readouterr().out
    assert not list(app.temporary_root.iterdir())
