"""Shared VAE reconstruction training, periodic evaluation, and epoch-boundary resume."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
import wave
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from Dataset.Preprocessing.company_data import (
    DEFAULT_SOURCE, DEFAULT_SPLIT_CSV, ChannelNormalizer, build_dataset, read_wav,
)
from Dataset.Preprocessing.dataset_splits import SPLIT_NAMES, split_recording_ids, write_split_csv
from device_support import resolve_device
from training_support import (
    atomic_json, atomic_torch_save, capture_rng_state, clone_model_state,
    prepare_run_directory, restore_rng_state, _replace_with_retry, _temporary_path,
)


DEFAULTS = {
    'epochs': 3, 'windows_per_recording': 16, 'batch_size': 8,
    'all_windows': False,
    'sequence_length': 2048, 'latent_dim': 16, 'learning_rate': .0003, 'beta': .001,
    'logvar_min': -8., 'logvar_max': 8.,
    'threads': 2, 'seed': 42, 'eval_every': 20, 'test_during_training': True,
    'device': 'auto',
}
LOCKED_SETTINGS = ('windows_per_recording', 'all_windows', 'batch_size', 'sequence_length', 'latent_dim',
                   'learning_rate', 'beta', 'logvar_min', 'logvar_max', 'seed')
LOSS_KEYS = ('loss', 'reconstruction_loss', 'kl_loss')


def evaluate(model, x, batch_size, beta, loss_function):
    """Deterministic posterior-mean evaluation; no gradients or latent sampling."""
    model.eval()
    device = next(model.parameters()).device
    sums = {key: 0. for key in LOSS_KEYS}
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = x[start:start+batch_size].to(device)
            values = loss_function(model(batch, sample=False), batch, beta)
            for key in sums:
                value = float(values[key])
                if not np.isfinite(value):
                    raise RuntimeError(f'Non-finite evaluation {key}')
                sums[key] += value*len(batch)
    return {key: value/len(x) for key, value in sums.items()}


def atomic_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        _replace_with_retry(temporary, path)
    finally:
        # Do not leave a stale temporary file if Windows keeps the target locked.
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def data_identity(dataset):
    """Compare the effective split and ordered model inputs, not CSV path names."""
    return {
        'active_split_rows': sorted(
            (row for row in dataset['split_rows'] if row['split'] in SPLIT_NAMES),
            key=lambda row: row['recording_id'],
        ),
        'recordings': sorted([
            {key: record[key] for key in ('recording_id', 'source_recording_id',
                                         'sample_rate', 'frame_count', 'window_count')}
            for record in dataset['manifest']
        ], key=lambda record: record['recording_id']),
        'ordered_windows': {split: dataset[split]['windows'] for split in SPLIT_NAMES},
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--split-csv', type=Path)
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='New empty run directory; default is a timestamped directory under runs/train')
    parser.add_argument('--resume', type=Path,
                        help='Full checkpoint to resume, or a legacy weights-only checkpoint to continue')
    parser.add_argument('--epochs', type=int, help='Total target epoch, including epochs already completed')
    parser.add_argument('--windows-per-recording', type=int)
    parser.add_argument('--all-windows', action=argparse.BooleanOptionalAction, default=None,
                        help='Use every complete non-overlapping window from each recording')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--sequence-length', type=int)
    parser.add_argument('--latent-dim', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--logvar-min', type=float)
    parser.add_argument('--logvar-max', type=float)
    parser.add_argument('--threads', type=int)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--eval-every', type=int, help='Validation/test evaluation interval; default 20')
    parser.add_argument('--test-during-training', action=argparse.BooleanOptionalAction, default=None,
                        help='Observe test metrics at evaluation epochs; only validation selects best (default: enabled)')
    return parser


def _resolve_settings(args, checkpoint, training_state):
    inherited = dict(DEFAULTS)
    inherited.update({'data_dir': str(DEFAULT_SOURCE), 'split_csv': str(DEFAULT_SPLIT_CSV)})
    if training_state is not None:
        inherited.update(training_state['settings'])
    elif checkpoint is not None:
        info = checkpoint.get('training_info', {})
        inherited['sequence_length'] = checkpoint['model_config']['sequence_length']
        inherited['latent_dim'] = checkpoint['model_config'].get('latent_dim', 16)
        inherited['epochs'] = int(info.get('epochs', 0))
        for key in ('seed', 'learning_rate', 'beta', 'batch_size'):
            if key in info:
                inherited[key] = info[key]
        for key in ('logvar_min', 'logvar_max'):
            if key in checkpoint.get('model_config', {}):
                inherited[key] = checkpoint['model_config'][key]
        if 'windows_per_recording_limit' in info:
            inherited['windows_per_recording'] = info['windows_per_recording_limit']
        if info.get('split_csv'):
            inherited['split_csv'] = info['split_csv']
    settings = dict(inherited)
    for key in (*DEFAULTS, 'data_dir', 'split_csv'):
        explicit = getattr(args, key)
        if explicit is None:
            continue
        if training_state is not None and key in LOCKED_SETTINGS and explicit != inherited[key]:
            raise ValueError(f'Full resume setting mismatch for {key}: saved {inherited[key]!r}, requested {explicit!r}')
        settings[key] = explicit
    for key in ('data_dir', 'split_csv'):
        settings[key] = str(Path(settings[key]).expanduser().resolve())
    for key in ('epochs', 'windows_per_recording', 'batch_size', 'sequence_length', 'latent_dim', 'threads', 'eval_every'):
        if key == 'windows_per_recording' and settings['all_windows']:
            settings[key] = None
            continue
        if settings[key] < 1:
            raise ValueError(f'{key} must be positive')
    if settings['learning_rate'] <= 0 or settings['beta'] < 0:
        raise ValueError('learning_rate must be positive and beta must be nonnegative')
    if (settings['logvar_min'] is None) != (settings['logvar_max'] is None):
        raise ValueError('logvar_min and logvar_max must be set together')
    if settings['logvar_min'] is not None and settings['logvar_min'] >= settings['logvar_max']:
        raise ValueError('logvar_min and logvar_max must be ordered')
    return settings


def _validate_legacy_identity(checkpoint, checkpoint_path, dataset):
    expected = checkpoint.get('training_info', {}).get('split_recordings')
    if expected is not None and expected != split_recording_ids(dataset['split_rows']):
        raise ValueError('Legacy checkpoint split assignments differ from the current CSV')
    saved_windows = checkpoint_path.parent/'window_manifest.json'
    if saved_windows.is_file():
        original = json.loads(saved_windows.read_text(encoding='utf-8-sig'))
        current = {split: dataset[split]['windows'] for split in SPLIT_NAMES}
        if original != current:
            raise ValueError('Legacy checkpoint ordered window manifest differs from current inputs')


def _same_weights(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)


def _history_files(output, history):
    atomic_json(history, output/'history.json')
    atomic_csv(output/'training_history.csv', history)


def _on_interrupt(output):
    latest = torch.load(output/'last.pt', map_location='cpu', weights_only=True)
    epoch = latest['training_state']['completed_epoch']
    atomic_json({'status': 'interrupted', 'completed_epoch': epoch,
                 'resume_checkpoint': str(output/'last.pt'),
                 'partial_epoch_saved': False}, output/'interruption.json')
    print(f'Training interrupted. Last completed epoch: {epoch}. Resume from {output / "last.pt"}', flush=True)
    return 130


def _engineering_outputs(output, dataset, tensors, model, config, initial_state, norm,
                         settings, loss_function, model_class, engine_class, streaming_class,
                         model_family, completed_epoch, start_epoch, cumulative_seconds,
                         previous_seconds, history, best_metric, best_epoch, initial_status,
                         baseline_kind, initial_epoch, resume_mode, resume_path):
    """Generate the existing final inference artifacts without affecting saved training RNG."""
    device = next(model.parameters()).device.type
    initial_model = model_class(config)
    initial_model.load_state_dict(initial_state)
    initial = engine_class(config, norm, initial_model, initial_status,
                           'fit_on_selected_train_windows_only', device=device)
    engine = engine_class(config, norm, model, 'smoke_trained_not_validated',
                         'fit_on_selected_train_windows_only', device=device)
    reloaded = engine_class.from_checkpoint(output/'final.pt', device=device)
    before = {split: evaluate(initial.model, tensors[split], settings['batch_size'],
                              settings['beta'], loss_function) for split in SPLIT_NAMES}
    after = {split: evaluate(reloaded.model, tensors[split], settings['batch_size'],
                             settings['beta'], loss_function) for split in SPLIT_NAMES}
    raw = dataset['test']['x'][:2]
    prediction = engine.predict_windows(raw)
    replay = reloaded.predict_windows(raw)
    reload_delta = float(np.max(np.abs(np.asarray(prediction['reconstruction_pcm'])-
                                      np.asarray(replay['reconstruction_pcm']))))
    if reload_delta > 1e-5:
        raise AssertionError('Checkpoint replay changed predictions')
    atomic_json(replay, output/'example_prediction.json')
    recording_id = dataset['test']['windows'][0]['recording_id']
    rate, real = read_wav(Path(settings['data_dir'])/recording_id/'audio_000000.wav')
    count = min(2, len(real)//config.sequence_length)
    frames = count*config.sequence_length
    with wave.open(str(output/'example_input.wav'), 'wb') as wav:
        wav.setnchannels(2); wav.setsampwidth(2); wav.setframerate(rate)
        wav.writeframes(real[:frames].astype('<i2').tobytes())
    atomic_json({'recording_id': recording_id, 'start_frame': 0, 'frame_count': frames,
                 'sample_rate_hz': rate, 'purpose': 'integration_example'}, output/'example_input_source.json')
    tail = min(137, config.sequence_length-1, len(real)-frames)
    stream_input = real[:frames+tail]
    stream = streaming_class(reloaded, batch_size=settings['batch_size'])
    streamed = []; cursor = 0
    for size in (17, 511, 3, 2048, 877, 10000):
        chunk = stream_input[cursor:cursor+size]
        cursor += len(chunk)
        streamed.extend(stream.push(chunk, rate))
    streamed.extend(stream.push(stream_input[cursor:], rate))
    direct = reloaded.predict_windows(real[:frames].reshape(count, config.sequence_length, 2),
                                       include_reconstruction=False)
    stream_error = float(np.max(np.abs(np.asarray([row['reconstruction_mse_standardized'] for row in streamed])-
                                      direct['reconstruction_mse_standardized'])))
    streaming_tolerance = 1e-4 if device == 'cuda' else 1e-5
    if len(streamed) != count or stream.pending_frames != tail or stream_error > streaming_tolerance:
        raise AssertionError('Streaming/offline check failed')
    sample_predictions = []
    for split in SPLIT_NAMES:
        for start in range(0, len(dataset[split]['x']), settings['batch_size']):
            batch = dataset[split]['x'][start:start+settings['batch_size']]
            baseline = initial.predict_windows(batch, include_reconstruction=False)
            trained = reloaded.predict_windows(batch, include_reconstruction=False)
            for j, window in enumerate(dataset[split]['windows'][start:start+settings['batch_size']]):
                row = {**window, 'split': split,
                       'initial_mse_standardized': baseline['reconstruction_mse_standardized'][j],
                       'trained_mse_standardized': trained['reconstruction_mse_standardized'][j],
                       'smoke_trained_mse_standardized': trained['reconstruction_mse_standardized'][j]}
                if initial_status == 'untrained':
                    row['untrained_mse_standardized'] = row['initial_mse_standardized']
                sample_predictions.append(row)
    atomic_csv(output/'window_predictions.csv', sample_predictions)
    probe = raw[:1]
    for _ in range(3):
        reloaded.predict_windows(probe, include_reconstruction=False)
    elapsed = []
    for _ in range(10):
        started = time.perf_counter()
        reloaded.predict_windows(probe, include_reconstruction=False)
        elapsed.append((time.perf_counter()-started)*1000)
    report = {
        'model_family': model_family, 'status': 'engineering_tests_passed',
        'validated_for_fault_detection': False, 'model_config': asdict(config),
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'torch_version': str(torch.__version__), 'device': str(reloaded.device), 'cpu_threads': settings['threads'],
        'sample_rate_hz': 44100, 'window_duration_ms': config.sequence_length/44100*1000,
        'recordings': len(dataset['manifest']), 'split_recordings': split_recording_ids(dataset['split_rows']),
        'split_csv': dataset['split_csv'], 'ignored_recordings': dataset['ignored_recordings'],
        'split_window_counts': {split: len(tensors[split]) for split in SPLIT_NAMES},
        'selected_signal_duration_seconds': sum(len(t) for t in tensors.values())*config.sequence_length/44100,
        'normalizer': norm.to_dict(), 'training_seconds': cumulative_seconds,
        'training_seconds_this_invocation': cumulative_seconds-previous_seconds,
        'timing_scope': 'Cumulative time of completed epochs, including scheduled evaluations, excluding checkpoint I/O and final engineering artifacts.',
        'epochs': completed_epoch, 'epochs_run_this_invocation': completed_epoch-start_epoch,
        'beta': settings['beta'], 'history': history, 'before': before, 'after': after,
        'initial_model_status': initial_status, 'baseline_kind': baseline_kind, 'initial_epoch': initial_epoch,
        'before_evaluation_timing': 'Initial-weight metrics, including test, are computed after training completes.',
        'resume_mode': resume_mode, 'resumed_from': str(resume_path) if resume_path else None,
        'eval_every': settings['eval_every'], 'test_during_training': settings['test_during_training'],
        'model_selection': 'Minimum validation reconstruction MSE; test measurements never select best.',
        'best_epoch': best_epoch, 'best_validation_reconstruction_mse': best_metric,
        'best_checkpoint': str(output/'best.pt'), 'final_checkpoint': str(output/'final.pt'),
        'mean_reconstruction_baseline_mse': {split: float((value**2).mean()) for split, value in tensors.items()},
        'verified_prediction_windows': len(sample_predictions),
        'checkpoint_max_abs_replay_delta_pcm': reload_delta,
        'streaming_check': {'emitted_windows': len(streamed), 'pending_frames': stream.pending_frames,
                            'max_score_delta': stream_error, 'tolerance': streaming_tolerance,
                            'comparison_note': 'CUDA uses a wider numerical tolerance for equivalent batch/stream inference.' if device == 'cuda' else 'CPU tolerance.'},
        'single_window_inference_median_ms': float(np.median(elapsed)),
        'single_window_inference_p95_ms': float(np.quantile(elapsed, .95)),
        'latency_measurement': f'Warm {reloaded.device}, batch=1, normalization and score/latent conversion included; WAV I/O and reconstruction JSON excluded.',
        'limitations': [
            'Reconstruction training has no calibrated fault threshold or supervised fault labels.',
            'Test measurements during training are observational only; final weights are the last epoch, while best.pt is selected using validation.',
            'Channels retain 44100 Hz and the observed 10 kHz component is not filtered.',
            'Source groups follow the CSV; window counts do not represent independent acquisition counts.',
            'Streaming windows are independent; LSTM hidden state restarts per window.',
            'Reconstruction loss and development-machine latency are not detection accuracy or deployment guarantees.',
        ],
    }
    atomic_json(report, output/'test_report.json')
    return report


def run_training(*, model_class, config_class, loss_function, engine_class,
                 streaming_class, model_family, argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    resume_path = args.resume.expanduser().resolve() if args.resume else None
    checkpoint = torch.load(resume_path, map_location='cpu', weights_only=True) if resume_path else None
    if checkpoint is not None and checkpoint['model_family'] != model_family:
        raise ValueError('Checkpoint belongs to a different model architecture')
    if checkpoint is not None and checkpoint['sample_rate_hz'] != 44100:
        raise ValueError('Checkpoint sampling rate does not match the 44100 Hz reconstruction task')
    previous_state = checkpoint.get('training_state') if checkpoint is not None else None
    settings = _resolve_settings(args, checkpoint, previous_state)
    config = (config_class(**checkpoint['model_config']) if checkpoint is not None else
              config_class(sequence_length=settings['sequence_length'], latent_dim=settings['latent_dim'],
                           logvar_min=settings['logvar_min'], logvar_max=settings['logvar_max']))
    if config.sequence_length != settings['sequence_length']:
        raise ValueError('Checkpoint model configuration differs from requested sequence_length')
    completed_epoch = (int(previous_state['completed_epoch']) if previous_state is not None else
                       int(checkpoint.get('training_info', {}).get('epochs', 0)) if checkpoint is not None else 0)
    if settings['epochs'] < completed_epoch:
        raise ValueError(f'Target --epochs must be at least completed epoch {completed_epoch}')
    output = prepare_run_directory(args.output_dir, model_family, 'reconstruction')
    window_limit = None if settings['all_windows'] else settings['windows_per_recording']
    dataset = build_dataset(settings['data_dir'], config.sequence_length,
                            window_limit, split_csv=settings['split_csv'])
    settings['split_csv'] = dataset['split_csv']
    identity = data_identity(dataset)
    if previous_state is not None and identity != previous_state['data_identity']:
        raise ValueError('Resume data identity mismatch: active CSV assignments, recording metadata, or ordered windows changed')
    if checkpoint is not None and previous_state is None:
        _validate_legacy_identity(checkpoint, resume_path, dataset)
    torch.set_num_threads(settings['threads'])
    device = resolve_device(settings['device'])
    print(f'Device: {device}', flush=True)
    random.seed(settings['seed']); np.random.seed(settings['seed']); torch.manual_seed(settings['seed'])
    model = model_class(config).to(device)
    if checkpoint is None:
        norm = ChannelNormalizer.fit(dataset['train']['x'])
    else:
        norm = ChannelNormalizer.from_dict(checkpoint['normalizer'])
        model.load_state_dict(checkpoint['state_dict'])
    tensors = {split: torch.from_numpy(norm.transform(dataset[split]['x'])) for split in SPLIT_NAMES}
    for split, tensor in tensors.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f'Non-finite input detected after normalization in split {split}')
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, threshold=1e-4,
        threshold_mode='rel', min_lr=1e-6,
    )
    generator = torch.Generator().manual_seed(settings['seed'])
    loader = DataLoader(TensorDataset(tensors['train']), batch_size=settings['batch_size'], shuffle=True,
                        generator=generator, num_workers=0)
    if previous_state is not None:
        optimizer.load_state_dict(previous_state['optimizer_state_dict'])
        if previous_state.get('scheduler_state_dict'):
            scheduler.load_state_dict(previous_state['scheduler_state_dict'])
        history = deepcopy(previous_state['history'])
        best_metric, best_epoch = previous_state['best_metric'], previous_state['best_epoch']
        best_state = deepcopy(previous_state['best_state_dict'])
        initial_state = deepcopy(previous_state['initial_state_dict'])
        initial_status = previous_state['initial_model_status']
        baseline_kind, initial_epoch = previous_state['baseline_kind'], previous_state['initial_epoch']
        cumulative_seconds = float(previous_state['cumulative_training_seconds'])
        resume_mode = 'full_state'
        print(f'Full-state resume after epoch {completed_epoch}.', flush=True)
    else:
        history = []; best_metric = None; best_epoch = None; best_state = None
        initial_state = clone_model_state(model)
        initial_status = checkpoint['model_status'] if checkpoint is not None else 'untrained'
        baseline_kind = 'continuation_start' if checkpoint is not None else 'random_initialization'
        initial_epoch = completed_epoch
        cumulative_seconds = 0.
        resume_mode = 'weights_only' if checkpoint is not None else 'new_training'
        if checkpoint is not None:
            print(f'Weights-only continuation / optimizer reset after epoch {completed_epoch}.', flush=True)
    start_epoch, previous_seconds = completed_epoch, cumulative_seconds
    training_info = {
        'purpose': 'reconstruction_training', 'seed': settings['seed'],
        'split_recordings': split_recording_ids(dataset['split_rows']),
        'split_csv': dataset['split_csv'], 'windows_per_recording_limit': None if settings['all_windows'] else settings['windows_per_recording'],
        'normalizer_fitted_to': 'training_windows_only', 'ignored_recordings': dataset['ignored_recordings'],
        'learning_rate': settings['learning_rate'], 'beta': settings['beta'], 'batch_size': settings['batch_size'],
        'logvar_min': settings['logvar_min'], 'logvar_max': settings['logvar_max'],
        'resume_mode': resume_mode, 'resumed_from': str(resume_path) if resume_path else None,
        'device': str(device),
    }

    def payload(epoch, weights=None, status=None, include_training_state=True):
        value = {
            'format_version': 1, 'model_family': model_family, 'model_config': asdict(config),
            'state_dict': weights if weights is not None else clone_model_state(model),
            'normalizer': norm.to_dict(), 'normalizer_provenance': 'fit_on_selected_train_windows_only',
            'model_status': status or ('untrained' if epoch == 0 and baseline_kind == 'random_initialization' else 'smoke_trained_not_validated'),
            'sample_rate_hz': 44100, 'training_info': {**training_info, 'epochs': epoch},
        }
        if include_training_state:
            value['training_state'] = {
                'schema_version': 1, 'completed_epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'rng_state': capture_rng_state({'loader': generator}, device=device),
                'history': history, 'best_metric': best_metric, 'best_epoch': best_epoch,
                'best_state_dict': best_state, 'initial_state_dict': initial_state,
                'initial_model_status': initial_status, 'baseline_kind': baseline_kind, 'initial_epoch': initial_epoch,
                'settings': dict(settings), 'data_identity': identity,
                'data_manifest': dataset['manifest'],
                'window_manifest': {split: dataset[split]['windows'] for split in SPLIT_NAMES},
                'split_rows': dataset['split_rows'], 'cumulative_training_seconds': cumulative_seconds,
            }
        return value

    atomic_json(dataset['manifest'], output/'data_manifest.json')
    atomic_json(dataset['ignored_recordings'], output/'ignored_recordings.json')
    atomic_json({split: dataset[split]['windows'] for split in SPLIT_NAMES}, output/'window_manifest.json')
    write_split_csv(output/'dataset_splits_used.csv', dataset['split_rows'])
    atomic_json({**settings, 'model_family': model_family, 'model_config': asdict(config),
                 'resume_mode': resume_mode, 'resumed_from': str(resume_path) if resume_path else None,
                 'best_selection': 'validation reconstruction MSE only'}, output/'training_config.json')
    baseline_payload = payload(initial_epoch, initial_state, initial_status, include_training_state=False)
    baseline_payload['training_info']['baseline_kind'] = baseline_kind
    atomic_torch_save(baseline_payload, output/'initial.pt')
    if initial_status == 'untrained':
        atomic_torch_save(baseline_payload, output/'untrained.pt')
    if best_state is not None:
        inherited_best_path = resume_path.parent/'best.pt'
        inherited_best = torch.load(inherited_best_path, map_location='cpu', weights_only=True) if inherited_best_path.is_file() else None
        if (inherited_best is not None and inherited_best.get('model_family') == model_family
                and asdict(config_class(**inherited_best['model_config'])) == asdict(config)
                and inherited_best.get('normalizer') == norm.to_dict()
                and _same_weights(inherited_best['state_dict'], best_state)):
            atomic_torch_save(inherited_best, output/'best.pt')
        else:
            # Older optimizer/RNG state is unavailable; retain an honest inference checkpoint.
            inherited = payload(best_epoch, best_state, 'smoke_trained_not_validated', include_training_state=False)
            inherited['training_info']['resume_capability'] = 'weights_only'
            atomic_torch_save(inherited, output/'best.pt')
    # Model/optimizer/loader creation and setup writes occur before RNG restoration.
    if previous_state is not None:
        restore_rng_state(previous_state['rng_state'], {'loader': generator}, device=device)
    atomic_torch_save(payload(completed_epoch), output/'last.pt')
    atomic_torch_save(payload(completed_epoch), output/'last_valid.pt')
    _history_files(output, history)
    print(f'Run directory: {output}', flush=True)
    try:
        for epoch in range(completed_epoch+1, settings['epochs']+1):
            started = time.perf_counter()
            model.train(); sums = {key: 0. for key in LOSS_KEYS}; count = 0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                terms = loss_function(model(batch), batch, settings['beta'])
                if any(not torch.isfinite(value).all() for value in terms.values()):
                    raise RuntimeError('Non-finite training loss component')
                terms['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                for key in sums:
                    sums[key] += float(terms[key].detach())*len(batch)
                count += len(batch)
            should_evaluate = epoch % settings['eval_every'] == 0 or epoch == settings['epochs']
            val = evaluate(model, tensors['val'], settings['batch_size'], settings['beta'], loss_function) if should_evaluate else None
            test = (evaluate(model, tensors['test'], settings['batch_size'], settings['beta'], loss_function)
                    if should_evaluate and settings['test_during_training'] else None)
            if val is not None:
                scheduler.step(val['reconstruction_loss'])
            improved = val is not None and (best_metric is None or val['reconstruction_loss'] < best_metric)
            if improved:
                best_metric, best_epoch = val['reconstruction_loss'], epoch
                best_state = clone_model_state(model)
            seconds = time.perf_counter()-started
            cumulative_seconds += seconds
            row = {'epoch': epoch, 'learning_rate': optimizer.param_groups[0]['lr'],
                   **{f'train_{key}': value/count for key, value in sums.items()},
                   **{f'val_{key}': val[key] if val is not None else None for key in LOSS_KEYS},
                   **{f'test_{key}': test[key] if test is not None else None for key in LOSS_KEYS},
                   'validation_evaluated': val is not None, 'test_evaluated': test is not None, 'seconds': seconds}
            history.append(row)
            completed_epoch = epoch
            saved = payload(completed_epoch)
            atomic_torch_save(saved, output/'last.pt')
            atomic_torch_save(saved, output/'last_valid.pt')
            if should_evaluate:
                atomic_torch_save(saved, output/'checkpoints'/f'epoch_{epoch:04d}.pt')
            if improved:
                atomic_torch_save(saved, output/'best.pt')
            _history_files(output, history)
            message = f'{model_family} epoch {epoch}/{settings["epochs"]}: train MSE={row["train_reconstruction_loss"]:.6f}'
            if val is not None:
                message += f'; val MSE={val["reconstruction_loss"]:.6f}'
            if test is not None:
                message += f'; test MSE={test["reconstruction_loss"]:.6f}'
            print(message, flush=True)
        if best_metric is None:
            # A completed weights-only checkpoint can be evaluated without an additional epoch.
            val = evaluate(model, tensors['val'], settings['batch_size'], settings['beta'], loss_function)
            best_metric, best_epoch = val['reconstruction_loss'], completed_epoch
            best_state = clone_model_state(model)
            atomic_torch_save(payload(completed_epoch), output/'best.pt')
        final_payload = payload(completed_epoch)
        atomic_torch_save(final_payload, output/'last.pt')
        atomic_torch_save(final_payload, output/'final.pt')
        atomic_torch_save(final_payload, output/'smoke_trained.pt')
        report = _engineering_outputs(
            output, dataset, tensors, model, config, initial_state, norm, settings,
            loss_function, model_class, engine_class, streaming_class, model_family,
            completed_epoch, start_epoch, cumulative_seconds, previous_seconds, history,
            best_metric, best_epoch, initial_status, baseline_kind, initial_epoch, resume_mode, resume_path,
        )
        print(json.dumps({'model': model_family, 'completed_epoch': completed_epoch,
                          'best_epoch': best_epoch, 'test_reconstruction_mse': report['after']['test']['reconstruction_loss'],
                          'report': str(output/'test_report.json')}, ensure_ascii=False), flush=True)
        return 0
    except KeyboardInterrupt:
        return _on_interrupt(output)
