"""Device routing contracts; CUDA failure tests do not require a GPU."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import device_support as devices


def test_explicit_cpu_never_initializes_cuda(monkeypatch):
    def forbidden():
        pytest.fail('Explicit CPU must not probe CUDA')

    monkeypatch.setattr(devices, '_probe_cuda', forbidden)
    assert devices.resolve_device('cpu') == torch.device('cpu')


def test_auto_cpu_build_falls_back_with_reason(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.version, 'cuda', None)
    assert devices.resolve_device() == torch.device('cpu')
    assert 'no CUDA support' in capsys.readouterr().err


@pytest.mark.parametrize('failure_point', ['allocation', 'synchronization'])
def test_auto_falls_back_after_a_real_probe_stage_fails(monkeypatch, capsys, failure_point):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    real_ones = torch.ones

    def allocate(*args, **kwargs):
        assert kwargs['device'] == 'cuda'
        if failure_point == 'allocation':
            raise RuntimeError('CUDA allocation failed')
        return real_ones(*args)

    def synchronize():
        raise RuntimeError('CUDA synchronization failed')

    monkeypatch.setattr(torch, 'ones', allocate)
    monkeypatch.setattr(torch.cuda, 'synchronize', synchronize)
    assert devices.resolve_device('auto') == torch.device('cpu')
    assert failure_point in capsys.readouterr().err


def test_explicit_cuda_reports_probe_failure_instead_of_using_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA device check failed'):
        devices.resolve_device('cuda')


def test_successful_probe_checks_allocation_math_and_synchronization(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    real_ones = torch.ones
    calls = []

    def allocate(*args, **kwargs):
        calls.append(('allocate', kwargs['device']))
        return real_ones(*args)

    monkeypatch.setattr(torch, 'ones', allocate)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: calls.append(('synchronize',)))
    assert devices.resolve_device('auto') == torch.device('cuda')
    assert calls == [('allocate', 'cuda'), ('synchronize',)]


def test_invalid_device_is_rejected():
    with pytest.raises(ValueError, match='auto, cpu, or cuda'):
        devices.resolve_device('mps')
