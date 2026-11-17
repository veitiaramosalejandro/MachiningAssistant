"""Run directories and resumable checkpoint utilities shared by trainers."""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import random
import time
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parent


def timestamped_training_root(output_root: str | Path) -> Path:
    stamp = datetime.now(ZoneInfo('Europe/Lisbon')).strftime('%Y-%m-%d_%H-%M-%S_%f')
    return Path(output_root).expanduser().resolve() / 'train' / stamp


def prepare_run_directory(output_dir, model_family: str, task: str) -> Path:
    if output_dir is None:
        model = {'CNN1D-VAE': 'cnn', 'LSTM-VAE': 'lstm'}.get(model_family, model_family)
        output = timestamped_training_root(PROJECT_ROOT / 'runs') / task / model
    else:
        output = Path(output_dir).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Training output is not empty: {output}. Choose a new directory; use --resume to load a checkpoint.')
    output.mkdir(parents=True, exist_ok=True)
    return output


def _temporary_path(path: Path) -> Path:
    """Create a per-process temporary name to avoid Windows collisions."""
    return path.with_name(f'{path.name}.{os.getpid()}.{time.time_ns()}.tmp')


def _replace_with_retry(temporary: Path, path: Path) -> None:
    """Replace a file, tolerating short antivirus/indexer locks on Windows."""
    last_error = None
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt == 7:
                raise
            time.sleep(0.15 * (attempt + 1))
    if last_error is not None:
        raise last_error


def atomic_torch_save(value, path: str | Path) -> None:
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    torch.save(value, temporary)
    _replace_with_retry(temporary, path)


def atomic_json(value, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    _replace_with_retry(temporary, path)


def capture_rng_state(generators=None, device=None) -> dict:
    import numpy as np
    import torch
    state = np.random.get_state()
    result = {
        'python': random.getstate(),
        'numpy': {'name': state[0], 'keys': state[1].tolist(), 'position': state[2],
                  'has_gauss': state[3], 'cached_gaussian': state[4]},
        'torch_cpu': torch.get_rng_state(),
        'generators': {name: generator.get_state() for name, generator in (generators or {}).items()},
    }
    if device is not None and torch.device(device).type == 'cuda':
        result['torch_cuda'] = [value.cpu() for value in torch.cuda.get_rng_state_all()]
    return result


def restore_rng_state(state: dict, generators=None, device=None) -> None:
    import numpy as np
    import torch
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state['name'], np.asarray(numpy_state['keys'], dtype=np.uint32),
                         numpy_state['position'], numpy_state['has_gauss'], numpy_state['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu'])
    if device is not None and torch.device(device).type == 'cuda' and state.get('torch_cuda'):
        torch.cuda.set_rng_state_all([value.cpu() for value in state['torch_cuda']])
    for name, generator in (generators or {}).items():
        generator.set_state(state['generators'][name])


def clone_model_state(model) -> dict:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
