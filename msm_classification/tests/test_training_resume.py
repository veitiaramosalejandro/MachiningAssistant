"""Small synthetic resume checks; no MSM files or production models are used."""
import copy
import json
import random
import sys

import numpy as np
import pytest
import torch

from test_training_splits import TinyConfig, TinyClassifier, assignments, metadata, tiny_loss
import experiment
import make_report
from Dataset.Preprocessing.dataset_splits import write_split_csv


class StochasticTiny(TinyClassifier):
    def forward(self, x, reconstruct=False):
        x = torch.nn.functional.dropout(x, .2, training=self.training)
        return super().forward(x, reconstruct=reconstruct)


def stochastic_loss(*args, **kwargs):
    result = tiny_loss(*args, **kwargs)
    multiplier = 1 + .01 * random.random() + .01 * np.random.random()
    return {key: value * multiplier for key, value in result.items()}


@pytest.fixture
def prepared_data(tmp_path):
    prepared = tmp_path/'prepared'
    prepared.mkdir()
    np.save(prepared/'windows.npy', np.random.default_rng(5).normal(size=(15, 4, 2)).astype(np.float32))
    np.save(prepared/'labels.npy', np.tile(np.arange(3), 5))
    (prepared/'windows.json').write_text(json.dumps(metadata()))
    (prepared/'preprocessing.json').write_text(json.dumps({
        'target_sample_rate_hz': 48000, 'sequence_length': 4,
        'windows_per_interval_limit': 1, 'label_edge_trim_seconds_each_side': .02,
    }))
    split_csv = tmp_path/'splits.csv'
    write_split_csv(split_csv, assignments())
    return prepared, split_csv


def run(monkeypatch, prepared_data, output, *, extra=(), loss=stochastic_loss, resume=None):
    prepared, split_csv = prepared_data
    argv = ['train', '--output', str(output)]
    if resume is None:
        argv += ['--prepared', str(prepared), '--split-csv', str(split_csv),
                 '--epochs', '4', '--eval-every', '2', '--batch-size', '2',
                 '--threads', '1', '--seed', '13', '--patience', '99']
    else:
        argv += ['--resume', str(resume)]
    monkeypatch.setattr(sys, 'argv', [*argv, *extra])
    return experiment.run_training(StochasticTiny, TinyConfig, loss, 'synthetic_test')


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def test_interrupted_full_resume_matches_continuous_training(monkeypatch, prepared_data, tmp_path):
    continuous = tmp_path/'continuous'
    run(monkeypatch, prepared_data, continuous)
    calls = 0

    def interrupt_after_half_of_epoch_three(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 6:
            raise KeyboardInterrupt()
        return stochastic_loss(*args, **kwargs)

    interrupted = tmp_path/'interrupted'
    with pytest.raises(SystemExit) as raised:
        run(monkeypatch, prepared_data, interrupted, loss=interrupt_after_half_of_epoch_three)
    assert raised.value.code == 130
    saved = load(interrupted/'last.pt')
    assert saved['training_state']['epoch'] == 2
    assert [row['epoch'] for row in saved['training_state']['history']] == [1, 2]
    assert not (interrupted/'final.pt').exists()
    resumed = tmp_path/'resumed'
    # No prepared/CSV/batch/thread/seed flags: all must come from the checkpoint.
    run(monkeypatch, prepared_data, resumed, resume=interrupted/'last.pt', extra=['--epochs', '4'])
    uninterrupted = load(continuous/'final.pt')
    restored = load(resumed/'final.pt')
    for key, value in uninterrupted['state_dict'].items():
        assert torch.equal(value, restored['state_dict'][key])
    a, b = uninterrupted['training_state'], restored['training_state']
    assert a['epoch'] == b['epoch'] == 4
    assert a['best_epoch'] == b['best_epoch']
    assert a['best_score'] == b['best_score']
    assert a['rng_state']['python'] == b['rng_state']['python']
    assert a['rng_state']['numpy'] == b['rng_state']['numpy']
    assert torch.equal(a['rng_state']['torch_cpu'], b['rng_state']['torch_cpu'])
    for name in ('train', 'val', 'test'):
        assert torch.equal(a['rng_state']['generators'][name], b['rng_state']['generators'][name])
    strip_time = lambda history: [{key: value for key, value in row.items() if key != 'seconds'} for row in history]
    assert strip_time(a['history']) == strip_time(b['history'])
    assert restored['normalizer'] == uninterrupted['normalizer']
    for name in ('last.pt', 'final.pt', 'best.pt', 'best_classifier.pt', 'checkpoints/epoch_0004.pt'):
        checkpoint = load(resumed/name)
        assert 'optimizer_state_dict' in checkpoint['training_state']
    best = load(resumed/'best.pt')
    assert best['training_state']['epoch'] == best['best_epoch']
    assert make_report.load_model('resumed', resumed)['report']['epochs_run'] == 4


def test_periodic_and_final_evaluations_are_sparse_and_test_is_default(monkeypatch, prepared_data, tmp_path):
    output = tmp_path/'periodic'
    run(monkeypatch, prepared_data, output, extra=['--epochs', '5'])
    history = json.loads((output/'history.json').read_text())
    assert [row['epoch'] for row in history if row['val_interval_macro_f1'] is not None] == [2, 4, 5]
    assert [row['epoch'] for row in history if row['test_interval_macro_f1'] is not None] == [2, 4, 5]
    assert sorted(path.name for path in (output/'checkpoints').glob('*.pt')) == ['epoch_0002.pt', 'epoch_0004.pt', 'epoch_0005.pt']
    report = make_report.load_model('periodic', output)['report']
    best_row = next(row for row in history if row['epoch'] == report['best_epoch'])
    assert best_row['val_interval_macro_f1'] == max(row['val_interval_macro_f1'] for row in history if row['val_interval_macro_f1'] is not None)
    assert report['test_during_training'] is True
    assert report['patience_unit'] == 'validation_evaluation_checks'
    assert load(output/'last.pt')['training_state']['epoch'] == 5
    invalid = copy.deepcopy(history)
    invalid[report['best_epoch'] - 1]['val_interval_macro_f1'] = None
    (output/'history.json').write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match='actual validation evaluation point'):
        make_report.load_model('invalid', output)


def test_fresh_default_cadence_and_optional_test_observation(monkeypatch, prepared_data, tmp_path):
    prepared, split_csv = prepared_data
    output = tmp_path/'default_cadence'
    monkeypatch.setattr(sys, 'argv', ['train', '--prepared', str(prepared), '--split-csv', str(split_csv),
                                     '--output', str(output), '--epochs', '21', '--threads', '1',
                                     '--no-test-during-training'])
    experiment.run_training(StochasticTiny, TinyConfig, stochastic_loss, 'synthetic_test')
    history = json.loads((output/'history.json').read_text())
    assert [row['epoch'] for row in history if row['val_interval_macro_f1'] is not None] == [20, 21]
    assert all(row['test_interval_macro_f1'] is None for row in history)
    assert (output/'checkpoints/epoch_0020.pt').is_file()
    assert (output/'checkpoints/epoch_0021.pt').is_file()
    assert 'test' in json.loads((output/'classification_report.json').read_text())


@pytest.mark.parametrize('change,match', [
    ('csv', 'active source-recording split'),
    ('order', 'ordered prepared-window identity'),
    ('optimizer', 'optimization settings'),
])
def test_full_resume_rejects_changed_active_identity_or_optimization(monkeypatch, prepared_data, tmp_path, change, match):
    source = tmp_path/'source'
    run(monkeypatch, prepared_data, source, extra=['--epochs', '2'])
    prepared, split_csv = prepared_data
    extra = ['--epochs', '4']
    if change == 'csv':
        write_split_csv(split_csv, assignments(train='dataset2', val='dataset1'))
    elif change == 'order':
        rows = metadata()
        rows[0], rows[1] = rows[1], rows[0]
        (prepared/'windows.json').write_text(json.dumps(rows))
    else:
        extra += ['--batch-size', '4']
    with pytest.raises(ValueError, match=match):
        run(monkeypatch, prepared_data, tmp_path/'destination', resume=source/'last.pt', extra=extra)


def test_resume_can_extend_epochs_and_change_evaluation_policy(monkeypatch, prepared_data, tmp_path):
    source = tmp_path/'source'
    run(monkeypatch, prepared_data, source, extra=['--epochs', '2'])
    output = tmp_path/'changed_policy'
    run(monkeypatch, prepared_data, output, resume=source/'last.pt',
        extra=['--epochs', '3', '--eval-every', '1', '--no-test-during-training'])
    state = load(output/'last.pt')['training_state']
    assert state['settings']['epochs'] == 3 and state['settings']['eval_every'] == 1
    assert state['settings']['test_during_training'] is False
    changes = state['resume_events'][-1]['settings_changes']
    assert {'epochs', 'eval_every', 'test_during_training'} <= set(changes)
    assert state['history'][-1]['val_interval_macro_f1'] is not None
    assert state['history'][-1]['test_interval_macro_f1'] is None


def test_nonempty_output_is_not_overwritten(monkeypatch, prepared_data, tmp_path):
    output = tmp_path/'occupied'
    output.mkdir()
    marker = output/'keep.txt'
    marker.write_text('preserve')
    with pytest.raises(FileExistsError, match='not empty'):
        run(monkeypatch, prepared_data, output)
    assert marker.read_text() == 'preserve'
    assert not (output/'last.pt').exists()


def test_legacy_classifier_continues_weights_with_explicit_provenance(monkeypatch, prepared_data, tmp_path, capsys):
    source = tmp_path/'source'
    run(monkeypatch, prepared_data, source, extra=['--epochs', '2'])
    legacy = load(source/'best_classifier.pt')
    del legacy['training_state']
    path = tmp_path/'legacy_best_classifier.pt'
    torch.save(legacy, path)
    prepared, split_csv = prepared_data
    output = tmp_path/'legacy_continuation'
    run(monkeypatch, prepared_data, output, resume=path,
        extra=['--prepared', str(prepared), '--split-csv', str(split_csv), '--epochs', '3', '--threads', '1'])
    report = make_report.load_model('legacy', output)['report']
    assert report['resume_kind'] == 'weights_only'
    assert report['start_epoch'] == legacy['best_epoch'] == 2
    assert report['epochs_executed_this_run'] == 1
    history = json.loads((output/'history.json').read_text())
    assert history[0]['resumed_anchor'] and history[0]['train_loss'] is None
    assert history[0]['val_interval_macro_f1'] is not None
    assert 'exact resume is unavailable' in capsys.readouterr().out
