"""Validate run preservation and exact RNG/state restoration without training."""
from datetime import datetime
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import training_support as support


def test_cpu_rng_capture_and_restore_do_not_touch_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU fallback must not access CUDA RNG')
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', forbidden)
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', forbidden)
    state = support.capture_rng_state(device='cpu')
    assert 'torch_cuda' not in state
    support.restore_rng_state(state, device='cpu')


def test_cuda_rng_is_saved_restored_and_optional_in_old_checkpoints(monkeypatch):
    cuda_state = [torch.tensor([1, 3, 7], dtype=torch.uint8)]
    restored = []
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: cuda_state)
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda values: restored.append(values))
    state = support.capture_rng_state(device='cuda')
    support.restore_rng_state(state, device='cuda')
    torch.testing.assert_close(restored[0][0], cuda_state[0])
    del state['torch_cuda']
    support.restore_rng_state(state, device='cuda')
    assert len(restored) == 1


def test_timestamp_uses_lisbon_time_and_does_not_create_directories(tmp_path, monkeypatch):
    seen_timezones = []
    timestamps = iter([(2026, 11, 10, 20, 1, 2, 123456), (2026, 11, 10, 20, 1, 2, 123457)])

    class FixedClock:
        @staticmethod
        def now(zone):
            seen_timezones.append(zone.key)
            return datetime(*next(timestamps), tzinfo=zone)

    monkeypatch.setattr(support, 'datetime', FixedClock)
    base = tmp_path/'run outputs'
    first = support.timestamped_training_root(base)
    second = support.timestamped_training_root(base)
    assert first == base.resolve()/'train'/'2026-11-10_20-01-02_123456'
    assert second == base.resolve()/'train'/'2026-11-10_20-01-02_123457'
    assert first != second
    assert seen_timezones == ['Europe/Lisbon', 'Europe/Lisbon']
    assert not base.exists()


@pytest.mark.parametrize('family,task,model', [
    ('CNN1D-VAE', 'reconstruction', 'cnn'), ('LSTM-VAE', 'msm', 'lstm'),
])
def test_default_run_directory_maps_task_and_model(tmp_path, monkeypatch, family, task, model):
    monkeypatch.setattr(support, 'PROJECT_ROOT', tmp_path)
    called = []
    base = tmp_path/'runs'/'train'/'fixed-time'

    def timestamp(output_root):
        called.append(Path(output_root))
        return base

    monkeypatch.setattr(support, 'timestamped_training_root', timestamp)
    result = support.prepare_run_directory(None, family, task)
    assert called == [tmp_path/'runs']
    assert result == base/task/model
    assert result.is_dir() and not list(result.iterdir())


def test_explicit_empty_directory_is_allowed_but_previous_results_are_preserved(tmp_path):
    target = tmp_path/'chosen run'
    target.mkdir()
    assert support.prepare_run_directory(target, 'CNN1D-VAE', 'reconstruction') == target.resolve()
    checkpoint = target/'last.pt'
    checkpoint.write_bytes(b'previous run must remain intact')
    with pytest.raises(FileExistsError, match='not empty'):
        support.prepare_run_directory(target, 'CNN1D-VAE', 'reconstruction')
    assert checkpoint.read_bytes() == b'previous run must remain intact'


def test_file_cannot_be_used_as_run_directory(tmp_path):
    path = tmp_path/'result-file'
    path.write_text('existing result')
    with pytest.raises(FileExistsError, match='not empty'):
        support.prepare_run_directory(path, 'LSTM-VAE', 'reconstruction')
    assert path.read_text() == 'existing result'


def test_atomic_torch_save_roundtrip_and_replacement(tmp_path):
    destination = tmp_path/'checkpoints'/'last.pt'
    support.atomic_torch_save({'epoch': 1, 'weight': torch.tensor([1.0])}, destination)
    support.atomic_torch_save({'epoch': 2, 'weight': torch.tensor([2.0])}, destination)
    state = torch.load(destination, map_location='cpu', weights_only=True)
    assert state['epoch'] == 2
    torch.testing.assert_close(state['weight'], torch.tensor([2.0]))
    assert not destination.with_name('last.pt.tmp').exists()


def test_failed_torch_serialization_keeps_the_previous_checkpoint(tmp_path, monkeypatch):
    destination = tmp_path/'last.pt'
    support.atomic_torch_save({'epoch': 7}, destination)

    def interrupted_save(value, path):
        Path(path).write_bytes(b'incomplete write')
        raise OSError('simulated interrupted checkpoint write')

    monkeypatch.setattr(torch, 'save', interrupted_save)
    with pytest.raises(OSError, match='interrupted checkpoint'):
        support.atomic_torch_save({'epoch': 8}, destination)
    assert torch.load(destination, map_location='cpu', weights_only=True)['epoch'] == 7


def test_atomic_json_roundtrip_and_invalid_value_preserves_previous_result(tmp_path):
    destination = tmp_path/'metrics'/'history.json'
    support.atomic_json({'epoch': 20, 'note': '验证'}, destination)
    assert json.loads(destination.read_text()) == {'epoch': 20, 'note': '验证'}
    assert not destination.with_name('history.json.tmp').exists()
    with pytest.raises(ValueError):
        support.atomic_json({'loss': float('nan')}, destination)
    assert json.loads(destination.read_text())['epoch'] == 20


def test_rng_checkpoint_restores_python_numpy_torch_and_loader_generator(tmp_path):
    original = support.capture_rng_state()
    try:
        random.seed(13); np.random.seed(17); torch.manual_seed(23)
        np.random.normal()  # exercise restoration of NumPy's cached Gaussian value
        loader = torch.Generator().manual_seed(31)
        snapshot = support.capture_rng_state({'loader': loader})
        destination = tmp_path/'rng.pt'
        support.atomic_torch_save({'rng_state': snapshot}, destination)
        expected_python = [random.random() for _ in range(4)]
        expected_numpy = np.random.normal(size=7)
        expected_torch = torch.rand(6)
        expected_loader = torch.randperm(19, generator=loader)
        random.seed(99); np.random.seed(99); torch.manual_seed(99); loader.manual_seed(99)
        loaded = torch.load(destination, map_location='cpu', weights_only=True)
        support.restore_rng_state(loaded['rng_state'], {'loader': loader})
        assert [random.random() for _ in range(4)] == expected_python
        np.testing.assert_array_equal(np.random.normal(size=7), expected_numpy)
        torch.testing.assert_close(torch.rand(6), expected_torch, rtol=0, atol=0)
        torch.testing.assert_close(torch.randperm(19, generator=loader), expected_loader, rtol=0, atol=0)
    finally:
        support.restore_rng_state(original)


def test_cloned_model_state_does_not_follow_future_parameter_or_buffer_updates():
    model = torch.nn.Linear(2, 1)
    model.register_buffer('running_value', torch.tensor([3.0]))
    snapshot = support.clone_model_state(model)
    expected_weight = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(100)
        model.running_value.add_(100)
    torch.testing.assert_close(snapshot['weight'], expected_weight, rtol=0, atol=0)
    torch.testing.assert_close(snapshot['running_value'], torch.tensor([3.0]), rtol=0, atol=0)
    assert all(value.device.type == 'cpu' and not value.requires_grad for value in snapshot.values())
