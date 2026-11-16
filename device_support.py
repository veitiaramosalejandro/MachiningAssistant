"""Choose CUDA only after a small operation succeeds; otherwise use CPU."""
from __future__ import annotations

import sys

import torch


def _probe_cuda() -> None:
    """Exercise allocation, a kernel and synchronization before selecting CUDA."""
    if not torch.cuda.is_available():
        if torch.version.cuda is None:
            raise RuntimeError('this PyTorch installation has no CUDA support')
        raise RuntimeError('no CUDA device is available to PyTorch')
    with torch.inference_mode():
        probe = torch.ones((8, 8), device='cuda')
        result = (probe @ probe.T) * 2 + 1
        torch.cuda.synchronize()
        if not bool(torch.isfinite(result).all().item()) or not bool((result == 17).all().item()):
            raise RuntimeError('the CUDA calculation returned an invalid result')


def resolve_device(requested='auto') -> torch.device:
    """Resolve auto/cpu/cuda without masking later model or memory errors.

    Auto falls back only when this initial CUDA probe fails. Explicit CUDA is
    strict so a requested GPU run cannot silently become a CPU run.
    """
    requested = str(requested).lower()
    if requested not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device must be auto, cpu, or cuda')
    if requested == 'cpu':
        return torch.device('cpu')
    try:
        _probe_cuda()
    except (RuntimeError, AssertionError, OSError) as exc:
        reason = str(exc).strip() or type(exc).__name__
        if requested == 'cuda':
            raise RuntimeError(f'CUDA device check failed: {reason}') from exc
        print(f'CUDA unavailable: {reason}. Using CPU.', file=sys.stderr, flush=True)
        return torch.device('cpu')
    return torch.device('cuda')
