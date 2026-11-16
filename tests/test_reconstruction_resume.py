"""Small real-CNN checks for epoch-boundary resume and checkpoint identity."""
from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'CNN1D-VAE'))
sys.path.insert(0, str(PROJECT))
import reconstruction_training as training
from Dataset.Preprocessing.dataset_splits import write_split_csv
from model import ModelConfig, VAE, loss_function
from runtime import StreamingPredictor, VAEEngine


@pytest.fixture
def recordings(tmp_path):
    data = tmp_path/'data'; data.mkdir()
    rows = [{'recording_id': name, 'source_recording_id': name, 'split': split}
            for name, split in [('train_a', 'train'), ('train_b', 'train'), ('validation', 'val'), ('testing', 'test')]]
    t = np.arange(64*6+5, dtype=np.float64)
    for i, row in enumerate(rows):
        folder = data/row['recording_id']; folder.mkdir()
        pcm = np.column_stack((350*np.sin(t*(.16+.01*i)),
                               900*np.cos(t*(.71+.02*i)))).astype('<i2')
        with wave.open(str(folder/'audio_000000.wav'), 'wb') as wav:
            wav.setnchannels(2); wav.setsampwidth(2); wav.setframerate(44100)
            wav.writeframes(pcm.tobytes())
        (folder/'metadata.json').write_text(json.dumps({'recording': row['recording_id']}))
    csv = tmp_path/'splits.csv'; write_split_csv(csv, rows)
    return data, csv, rows


def invoke(output, recordings, epochs, resume=None, extra=(), custom_loss=loss_function):
    argv = ['--output-dir', str(output), '--epochs', str(epochs)]
    if resume is None:
        data, csv, _ = recordings
        argv += ['--data-dir', str(data), '--split-csv', str(csv), '--sequence-length', '64',
                 '--windows-per-recording', '2', '--batch-size', '2', '--threads', '1',
                 '--seed', '37', '--eval-every', '2']
    else:
        argv += ['--resume', str(resume)]
    argv += list(extra)
    return training.run_training(model_class=VAE, config_class=ModelConfig,
                                 loss_function=custom_loss, engine_class=VAEEngine,
                                 streaming_class=StreamingPredictor, model_family='CNN1D-VAE', argv=argv)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def test_four_epochs_match_two_then_resume_in_model_adam_rng_and_history(tmp_path, recordings):
    full, first, resumed = [tmp_path/name for name in ('full', 'first', 'resumed')]
    assert invoke(full, recordings, 4) == 0
    assert invoke(first, recordings, 2) == 0
    assert invoke(resumed, recordings, 4, resume=first/'last.pt') == 0
    a, b = load(full/'final.pt'), load(resumed/'final.pt')
    assert_nested_equal(a['state_dict'], b['state_dict'])
    for key in ('optimizer_state_dict', 'rng_state', 'initial_state_dict', 'best_state_dict'):
        assert_nested_equal(a['training_state'][key], b['training_state'][key])
    for old, new in zip(a['training_state']['history'], b['training_state']['history']):
        assert {k: v for k, v in old.items() if k != 'seconds'} == {k: v for k, v in new.items() if k != 'seconds'}
    assert b['training_state']['completed_epoch'] == 4
    assert b['training_state']['settings']['batch_size'] == 2
    assert b['training_state']['settings']['windows_per_recording'] == 2
    assert b['training_state']['settings']['threads'] == 1
    assert b['training_state']['initial_model_status'] == 'untrained'
    assert b['training_info']['resume_mode'] == 'full_state'


def test_periodic_best_last_and_final_checkpoints_load_for_inference(tmp_path, recordings):
    output = tmp_path/'run'
    invoke(output, recordings, 4)
    history = json.loads((output/'history.json').read_text())
    assert [r['epoch'] for r in history if r['validation_evaluated']] == [2, 4]
    assert [r['epoch'] for r in history if r['test_evaluated']] == [2, 4]
    assert history[0]['val_reconstruction_loss'] is None
    assert history[2]['test_reconstruction_loss'] is None
    assert sorted(p.name for p in (output/'checkpoints').glob('*.pt')) == ['epoch_0002.pt', 'epoch_0004.pt']
    winner = min((r for r in history if r['validation_evaluated']), key=lambda r: r['val_reconstruction_loss'])
    best = load(output/'best.pt')
    assert best['training_state']['completed_epoch'] == winner['epoch']
    assert best['training_state']['best_metric'] == winner['val_reconstruction_loss']
    signal = np.zeros((1, 64, 2), dtype=np.float32)
    for path in [output/'last.pt', output/'best.pt', output/'final.pt',
                 output/'checkpoints/epoch_0002.pt', output/'checkpoints/epoch_0004.pt']:
        prediction = VAEEngine.from_checkpoint(path).predict_windows(signal)
        assert np.isfinite(prediction['reconstruction_mse_standardized']).all()
    assert_nested_equal(load(output/'final.pt')['state_dict'], load(output/'smoke_trained.pt')['state_dict'])


def test_full_resume_does_not_refit_normalizer(tmp_path, recordings, monkeypatch):
    first = tmp_path/'first'; invoke(first, recordings, 2)

    def forbidden_fit(*args, **kwargs):
        raise AssertionError('Normalizer must not be fitted during resume')

    monkeypatch.setattr(training.ChannelNormalizer, 'fit', forbidden_fit)
    assert invoke(tmp_path/'resumed', recordings, 3, resume=first/'last.pt') == 0
    assert load(first/'last.pt')['normalizer'] == load(tmp_path/'resumed/last.pt')['normalizer']


def test_legacy_default_config_fields_do_not_discard_best_resume_state(tmp_path, recordings):
    first = tmp_path/'first'
    invoke(first, recordings, 2)
    for name in ('last.pt', 'best.pt'):
        checkpoint = load(first/name)
        checkpoint['model_config'].pop('logvar_min', None)
        checkpoint['model_config'].pop('logvar_max', None)
        torch.save(checkpoint, first/name)
    output = tmp_path/'resumed'
    assert invoke(output, recordings, 2, resume=first/'last.pt') == 0
    retained = load(output/'best.pt')
    assert retained['training_state']['completed_epoch'] == 2
    assert_nested_equal(retained['training_state']['optimizer_state_dict'],
                        load(first/'best.pt')['training_state']['optimizer_state_dict'])


def test_legacy_continuation_resets_optimizer_and_keeps_trained_initial_identity(tmp_path, recordings, capsys):
    first = tmp_path/'first'; invoke(first, recordings, 2)
    legacy = load(first/'final.pt'); legacy.pop('training_state')
    legacy_path = tmp_path/'legacy.pt'; torch.save(legacy, legacy_path)
    output = tmp_path/'continued'
    assert invoke(output, recordings, 3, resume=legacy_path, extra=['--data-dir', str(recordings[0]), '--threads', '1']) == 0
    result = load(output/'final.pt')
    assert 'Weights-only continuation / optimizer reset' in capsys.readouterr().out
    assert result['training_state']['completed_epoch'] == 3
    assert len(result['training_state']['history']) == 1
    assert result['training_state']['baseline_kind'] == 'continuation_start'
    assert_nested_equal(result['training_state']['initial_state_dict'], legacy['state_dict'])
    assert (output/'initial.pt').exists() and not (output/'untrained.pt').exists()
    assert all(float(s['step']) == 2 for s in result['training_state']['optimizer_state_dict']['state'].values())
    report = json.loads((output/'test_report.json').read_text())
    assert report['initial_epoch'] == 2 and report['baseline_kind'] == 'continuation_start'


def test_nonempty_output_is_not_overwritten(tmp_path, recordings):
    output = tmp_path/'existing'; output.mkdir()
    marker = output/'keep.txt'; marker.write_text('saved result')
    with pytest.raises(FileExistsError, match='not empty'):
        invoke(output, recordings, 2)
    assert marker.read_text() == 'saved result'
    assert not (output/'last.pt').exists()


@pytest.mark.parametrize('change', ['assignment', 'order'])
def test_resume_rejects_changed_csv_effective_inputs(tmp_path, recordings, change):
    first = tmp_path/'first'; invoke(first, recordings, 2)
    _, csv, original_rows = recordings
    rows = [dict(row) for row in original_rows]
    if change == 'assignment':
        rows[0]['split'], rows[2]['split'] = 'val', 'train'
    else:
        rows[0], rows[1] = rows[1], rows[0]
    write_split_csv(csv, rows)
    with pytest.raises(ValueError, match='data identity mismatch'):
        invoke(tmp_path/'resumed', recordings, 4, resume=first/'last.pt')


def test_full_resume_rejects_training_setting_change(tmp_path, recordings):
    first = tmp_path/'first'; invoke(first, recordings, 2)
    with pytest.raises(ValueError, match='setting mismatch for batch_size'):
        invoke(tmp_path/'resumed', recordings, 4, resume=first/'last.pt', extra=['--batch-size', '4'])


def test_interrupt_keeps_epoch_zero_then_resumes_exactly(tmp_path, recordings):
    full, interrupted, resumed = [tmp_path/name for name in ('full', 'interrupted', 'resumed')]
    invoke(full, recordings, 2)
    calls = 0

    def interrupted_loss(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt()
        return loss_function(*args, **kwargs)

    assert invoke(interrupted, recordings, 2, custom_loss=interrupted_loss) == 130
    state = load(interrupted/'last.pt')
    assert state['training_state']['completed_epoch'] == 0
    assert state['training_state']['optimizer_state_dict']['state'] == {}
    assert_nested_equal(state['state_dict'], state['training_state']['initial_state_dict'])
    assert not (interrupted/'final.pt').exists()
    assert invoke(resumed, recordings, 2, resume=interrupted/'last.pt') == 0
    assert_nested_equal(load(full/'final.pt')['state_dict'], load(resumed/'final.pt')['state_dict'])
    assert_nested_equal(load(full/'final.pt')['training_state']['rng_state'], load(resumed/'final.pt')['training_state']['rng_state'])


def test_final_epoch_is_evaluated_and_test_observation_can_be_disabled(tmp_path, recordings):
    output = tmp_path/'run'
    invoke(output, recordings, 3, extra=['--no-test-during-training'])
    history = json.loads((output/'history.json').read_text())
    assert [r['epoch'] for r in history if r['validation_evaluated']] == [2, 3]
    assert not any(r['test_evaluated'] for r in history)
    report = json.loads((output/'test_report.json').read_text())
    assert np.isfinite(report['after']['test']['reconstruction_loss'])
