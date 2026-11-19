"""Supervised classification of complete raw WAV samples with VAE encoders.

Class names come from labels.json, and each run rereads train/val/test CSVs.
The module name is retained for the original four-class dataset; class count
and label names are configurable. The public CLI always uses full five-second
signals; valid_frames is injectable only through the Python test interface.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from Dataset.Preprocessing.fourclass_data import (
    DEFAULT_DATA_DIR, SAMPLE_RATE, VALID_FRAMES, RawFourClassDataset,
    fit_normalizer, load_class_names, load_manifest, model_input_length,
)
from Dataset.Preprocessing.cnc_features import CncFeatureNormalizer, FEATURE_NAMES
from Dataset.Preprocessing.active_dataset import resolve_dataset_dir
from device_support import resolve_device
from training_support import (
    atomic_json, atomic_torch_save, capture_rng_state, clone_model_state,
    prepare_run_directory, restore_rng_state,
)


SPLITS = ('train', 'val', 'test')
DEFAULTS = {
    'epochs': 20,
    'batch_size': 1,
    'learning_rate': .0001,
    'reconstruction_weight': .05,
    'beta': .001,
    'gradient_clip': 5.,
    'threads': 2,
    'seed': 42,
    'eval_every': 20,
    'test_during_training': True,
    'device': 'auto',
    'context_embedding_dim': 8,
    'cnc_context': None,
}
LOCKED_SETTINGS = (
    'data_dir',
    'batch_size',
    'learning_rate',
    'reconstruction_weight',
    'beta',
    'gradient_clip',
    'seed',
    'threads',
    'context_embedding_dim',
    'cnc_context',
)


def load_cnc_contexts(path):
    """Load raw service contexts keyed by sample_id from a JSON sidecar.

    Accepted forms are ``{"sample_id": {context...}}`` and a list of
    ``{"sample_id": ..., "context": {...}}`` records. The sidecar is an
    offline training artifact; live services should send the same context
    contract before inference.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'CNC context JSON not found: {path}')
    payload = json.loads(path.read_text(encoding='utf-8-sig'))
    if isinstance(payload, dict) and isinstance(payload.get('records'), list):
        records = payload['records']
    elif isinstance(payload, dict):
        records = [{'sample_id': sample_id, 'context': context}
                   for sample_id, context in payload.items()]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError('CNC context JSON must be a mapping or a records list')
    contexts = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get('sample_id'), str):
            raise ValueError('Each CNC context record needs a string sample_id')
        if record['sample_id'] in contexts:
            raise ValueError(f'Duplicate CNC context sample_id: {record["sample_id"]}')
        context = record.get('context', record)
        if not isinstance(context, dict):
            raise ValueError(f'CNC context for {record["sample_id"]} must be an object')
        contexts[record['sample_id']] = context
    if not contexts:
        raise ValueError('CNC context JSON contains no records')
    return contexts


def _batch_parts(batch):
    if len(batch) == 2:
        signal, labels = batch
        return signal, None, labels
    if len(batch) == 3:
        signal, context, labels = batch
        return signal, context, labels
    raise ValueError('Classification batch must contain signal, optional context and labels')


def joint_loss(output, labels, signal, valid_frames, reconstruction_weight=.05, beta=.001):
    """Only valid PCM frames contribute to reconstruction MSE; CNN padding does not."""
    ce = F.cross_entropy(output['logits'], labels)
    mse = ce.new_zeros(())
    if 'reconstruction' in output:
        if output['reconstruction'].shape != signal.shape:
            raise ValueError('Reconstruction and input tensor shapes differ')
        mse = F.mse_loss(output['reconstruction'][:, :valid_frames], signal[:, :valid_frames])
    elif reconstruction_weight > 0:
        raise ValueError('Positive reconstruction_weight requires a reconstruction output')
    mu, logvar = output['mu'], output['logvar']
    kl = (-.5*(1 + logvar - mu.square() - logvar.exp()).sum(dim=-1)).mean()
    return {'loss': ce + reconstruction_weight*mse + beta*kl,
            'ce_loss': ce, 'reconstruction_loss': mse, 'kl_loss': kl}


def clip_gradients(parameters, max_norm):
    """Clip finite gradients even when their float32 norm reduction overflows.

    The normal path retains PyTorch's clipping behavior. The fallback computes
    only the reduction in float64; actual NaN/Inf gradients still stop the run.
    """
    parameters = list(parameters)
    try:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    except RuntimeError:
        gradients = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
        if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise
        norms = torch.stack([torch.linalg.vector_norm(gradient, dtype=torch.float64) for gradient in gradients])
        total_norm = torch.linalg.vector_norm(norms)
        if not torch.isfinite(total_norm):
            raise
        coefficient = min(1., float(max_norm / (total_norm + 1e-6)))
        with torch.no_grad():
            for gradient in gradients:
                gradient.mul_(coefficient)
        return total_norm


def classification_metrics(truth, predicted, class_names):
    classes = len(class_names)
    matrix = np.zeros((classes, classes), dtype=np.int64)
    truth = np.asarray(truth, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    np.add.at(matrix, (truth, predicted), 1)
    diagonal = matrix.diagonal().astype(np.float64)
    support = matrix.sum(axis=1)
    precision = np.divide(diagonal, matrix.sum(axis=0), out=np.zeros(classes), where=matrix.sum(axis=0) != 0)
    recall = np.divide(diagonal, support, out=np.zeros(classes), where=support != 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(classes), where=precision+recall != 0)
    return {'accuracy': float(diagonal.sum()/len(truth)),
            'macro_precision': float(precision.mean()), 'macro_recall': float(recall.mean()),
            'macro_f1': float(f1.mean()),
            'balanced_accuracy': float(recall[support > 0].mean()), 'sample_count': int(len(truth)),
            'confusion_matrix': matrix.tolist(),
            'per_class': [{'class_id': index, 'class_name': name, 'precision': float(precision[index]),
                           'recall': float(recall[index]), 'f1': float(f1[index]), 'support': int(support[index])}
                          for index, name in enumerate(class_names)]}


def complete_metrics(metrics):
    """Add P/R to older reports from their saved confusion matrix, in place."""
    if metrics is None:
        return None
    if 'macro_precision' not in metrics or 'macro_recall' not in metrics:
        matrix = np.asarray(metrics['confusion_matrix'], dtype=np.float64)
        diagonal = matrix.diagonal()
        predicted_counts, true_counts = matrix.sum(axis=0), matrix.sum(axis=1)
        precision = np.divide(diagonal, predicted_counts, out=np.zeros(len(matrix)), where=predicted_counts != 0)
        recall = np.divide(diagonal, true_counts, out=np.zeros(len(matrix)), where=true_counts != 0)
        metrics.setdefault('macro_precision', float(precision.mean()))
        metrics.setdefault('macro_recall', float(recall.mean()))
    return metrics


def restore_history_metrics(history, evaluations):
    """Keep old checkpoint histories usable when the CSV gains P/R columns."""
    by_epoch = {}
    for entry in evaluations:
        by_epoch[entry['epoch']] = {
            'val': complete_metrics(entry.get('validation')),
            'test': complete_metrics(entry.get('test')),
        }
    for row in history:
        for split in ('val', 'test'):
            metrics = by_epoch.get(row['epoch'], {}).get(split)
            for name in ('macro_precision', 'macro_recall'):
                row[f'{split}_{name}'] = metrics[name] if metrics is not None else row.get(f'{split}_{name}')


def format_classification_metrics(metrics):
    """Use the same named aggregate metrics in training and independent tests."""
    return (f'ACC={metrics["accuracy"]:.6f}; P(macro)={metrics["macro_precision"]:.6f}; '
            f'R(macro)={metrics["macro_recall"]:.6f}; F1(macro)={metrics["macro_f1"]:.6f}')


def section_heading(title):
    return f' {title} '.center(64, '-')


def evaluate(model, loader, class_names):
    model.eval()
    device = next(model.parameters()).device
    truths, scores = [], []
    with torch.inference_mode():
        for batch in loader:
            signal, context, labels = _batch_parts(batch)
            # Classification uses mu; eval mode also disables latent sampling.
            kwargs = {} if context is None else {'context': context.to(device)}
            logits = model(signal.to(device), reconstruct=False, **kwargs)['logits']
            probabilities = logits.softmax(dim=-1)
            if not torch.isfinite(probabilities).all():
                raise RuntimeError('Non-finite class scores')
            truths.append(labels.cpu().numpy()); scores.append(probabilities.cpu().numpy())
    truth, probabilities = np.concatenate(truths), np.concatenate(scores)
    return classification_metrics(truth, probabilities.argmax(axis=1), class_names), truth, probabilities


def data_identity(data_dir, rows, class_names, valid_frames):
    """Record selected sample identities and WAV metadata, without signal hashing."""
    identity_rows = []
    for row in rows:
        stat = (data_dir/row['wav_file']).stat()
        identity_rows.append({**row, 'wav_bytes': stat.st_size, 'wav_mtime_ns': stat.st_mtime_ns})
    return {'data_dir': str(data_dir), 'class_names': list(class_names),
            'valid_frames': valid_frames, 'ordered_samples': identity_rows}


def _write_csv(path, rows):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def _save_history(output, history, evaluations):
    atomic_json(history, output/'history.json')
    atomic_json(evaluations, output/'evaluations.json')
    if history:
        _write_csv(output/'training_history.csv', history)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, help='Dataset/WAV root; defaults to the --csv parent or the active dataset')
    parser.add_argument('--csv', type=Path, help='One combined samples.csv with a split column for train/val/test')
    parser.add_argument('--output-dir', type=Path, help='New empty run directory; default: runs/train/<timestamp>/classification/<model>')
    parser.add_argument('--resume', type=Path, help='Matching full classification checkpoint; reconstruction/MSM checkpoints are not accepted')
    parser.add_argument('--cnc-context', type=Path,
                        help='JSON sidecar with one raw MachiningContext per sample_id')
    parser.add_argument('--context-embedding-dim', type=int,
                        help='CNC MLP embedding width for multimodal CNN1D-VAE; default 8')
    parser.add_argument('--epochs', type=int, help='Total target epoch, including completed epochs; default 20')
    parser.add_argument('--batch-size', type=int, help='Default 1 for full five-second waveforms')
    parser.add_argument('--lstm-frame-size', type=int,
                        help='LSTM only: raw sample frames per recurrent step; fresh default 441 (10 ms); 1 keeps the legacy layout. Ignored by CNN.')
    parser.add_argument('--learning-rate', '--lr', dest='learning_rate', type=float,
                        help='Fresh-run default 0.0001; a full resume retains its saved learning rate')
    parser.add_argument('--reconstruction-weight', type=float)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--gradient-clip', type=float)
    parser.add_argument('--threads', type=int)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'),
                        help='Default auto: use CUDA after a successful compute probe, otherwise CPU')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--eval-every', type=int, help='Evaluate validation and test every N epochs; default 20; also evaluate the final epoch')
    parser.add_argument('--test-during-training', action=argparse.BooleanOptionalAction, default=None,
                        help='Observe periodic test scores without using them to select best; default enabled')
    return parser


def _settings(args, previous):
    settings = {**DEFAULTS, 'data_dir': str(DEFAULT_DATA_DIR.resolve()), 'csv': None}
    if previous is not None:
        settings.update(previous['settings'])
    elif args.data_dir is None:
        settings['data_dir'] = str(args.csv.expanduser().resolve().parent if args.csv is not None else resolve_dataset_dir())
    for key in settings:
        requested = getattr(args, key)
        if requested is None:
            continue
        if key in ('data_dir', 'csv', 'cnc_context') and requested is not None:
            requested = str(Path(requested).expanduser().resolve())
        if previous is not None and key == 'csv' and requested != settings['csv']:
            raise ValueError('Resume CSV differs from the saved input. Start a new run to select a different CSV.')
        if previous is not None and key in LOCKED_SETTINGS and requested != settings[key]:
            raise ValueError(f'Resume setting mismatch for {key}: saved {settings[key]!r}, requested {requested!r}')
        settings[key] = requested
    settings['data_dir'] = str(Path(settings['data_dir']).expanduser().resolve())
    if settings['csv'] is not None:
        settings['csv'] = str(Path(settings['csv']).expanduser().resolve())
    if settings['cnc_context'] is not None:
        settings['cnc_context'] = str(Path(settings['cnc_context']).expanduser().resolve())
    for key in ('epochs', 'batch_size', 'threads', 'eval_every'):
        if settings[key] < 1:
            raise ValueError(f'{key} must be positive')
    for key in ('learning_rate', 'gradient_clip'):
        if not np.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('reconstruction_weight', 'beta'):
        if not np.isfinite(settings[key]) or settings[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if settings.get('context_embedding_dim', 8) < 1:
        raise ValueError('context_embedding_dim must be positive')
    return settings


def run_training(StateClassifier, ModelConfig, model_family, argv=None, *, valid_frames=VALID_FRAMES):
    args = _parser().parse_args(argv)
    if args.lstm_frame_size is not None and args.lstm_frame_size < 1:
        raise ValueError('lstm_frame_size must be a positive integer')
    resume_path = args.resume.expanduser().resolve() if args.resume is not None else None
    checkpoint = None
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f'Resume checkpoint not found: {resume_path}')
        checkpoint = torch.load(resume_path, map_location='cpu', weights_only=True)
        if checkpoint.get('task') != 'classification' or checkpoint.get('model_family') != model_family:
            raise ValueError('Resume requires a matching classification checkpoint, not reconstruction or MSM weights')
        if not isinstance(checkpoint.get('training_state'), dict):
            raise ValueError('Checkpoint has no complete classification training state; weights-only continuation is unsupported')
        if checkpoint.get('sample_rate') != SAMPLE_RATE or checkpoint.get('valid_frames') != valid_frames:
            raise ValueError('Resume sample rate or valid signal length differs')
    previous = checkpoint['training_state'] if checkpoint is not None else None
    settings = _settings(args, previous)
    data_dir = Path(settings['data_dir'])
    if settings['csv'] is not None and not Path(settings['csv']).is_file():
        raise FileNotFoundError(f'Combined CSV file not found: {settings["csv"]}')
    class_names = list(load_class_names(data_dir))
    rows = load_manifest(data_dir, valid_frames=valid_frames, csv_path=settings['csv'])
    subsets = {split: [row for row in rows if row['split'] == split] for split in SPLITS}
    if any(not values for values in subsets.values()):
        raise ValueError('Training, validation and test CSVs must all be nonempty')
    train_counts = np.bincount([row['label_index'] for row in subsets['train']], minlength=len(class_names))
    absent = [name for name, count in zip(class_names, train_counts) if count == 0]
    if absent:
        raise ValueError(f'Configured classes missing from train.csv: {absent}')
    input_length = model_input_length(model_family, valid_frames=valid_frames)
    raw_contexts = None
    context_vectors = None
    context_normalizer = None
    context_dim = 0
    context_embedding_dim = int(settings.get('context_embedding_dim', 8))
    if settings.get('cnc_context') is not None:
        if model_family != 'CNN1D-VAE':
            raise ValueError('CNC multimodal input is currently supported only for CNN1D-VAE')
        raw_contexts = load_cnc_contexts(settings['cnc_context'])
        sample_ids = {row['sample_id'] for row in rows}
        if set(raw_contexts) != sample_ids:
            missing = sorted(sample_ids - set(raw_contexts))
            unknown = sorted(set(raw_contexts) - sample_ids)
            if missing:
                raise ValueError(f'Missing CNC context for sample_id: {missing[0]}')
            raise ValueError(f'CNC context contains unknown sample_id: {unknown[0]}')
        context_dim = len(FEATURE_NAMES)
    identity = data_identity(data_dir, rows, class_names, valid_frames)
    if checkpoint is not None:
        if checkpoint.get('class_names') != class_names or checkpoint.get('num_classes') != len(class_names):
            raise ValueError('Resume class names, order or count differs from labels.json')
        if previous['data_identity'] != identity:
            raise ValueError('Resume data identity changed: selected rows, labels, split membership or WAV files differ')
        config = ModelConfig(**checkpoint['model_config'])
        if config.sequence_length != input_length or config.input_channels != 2:
            raise ValueError('Resume model configuration does not match complete-signal inputs')
        if (model_family == 'LSTM-VAE' and args.lstm_frame_size is not None
                and args.lstm_frame_size != config.frame_size):
            raise ValueError('Resume frame_size differs from the checkpoint. Framing changes the model architecture; start a new run without --resume.')
        normalizer = deepcopy(checkpoint['normalizer'])
        saved_context_dim = int(checkpoint.get('context_dim', 0))
        if saved_context_dim != context_dim:
            raise ValueError('Resume context configuration differs from the checkpoint')
        context_embedding_dim = int(checkpoint.get('context_embedding_dim', context_embedding_dim))
        saved_context_normalizer = checkpoint.get('context_normalizer')
        if context_dim and not isinstance(saved_context_normalizer, dict):
            raise ValueError('Multimodal checkpoint lacks context_normalizer')
    else:
        config_values = dict(sequence_length=input_length, input_channels=2, latent_dim=16,
                             logvar_min=-10., logvar_max=10.)
        if model_family == 'LSTM-VAE':
            config_values['frame_size'] = args.lstm_frame_size if args.lstm_frame_size is not None else 441
        config = ModelConfig(**config_values)
        normalizer = fit_normalizer(data_dir, subsets['train'])
        saved_context_normalizer = None
    if context_dim:
        if saved_context_normalizer is not None:
            context_normalizer = CncFeatureNormalizer.from_dict(saved_context_normalizer)
        else:
            context_normalizer = CncFeatureNormalizer.fit_contexts(
                [raw_contexts[row['sample_id']] for row in subsets['train']]
            )
        context_vectors = {
            sample_id: context_normalizer.transform_context(context)
            for sample_id, context in raw_contexts.items()
        }
    completed_epoch = int(previous['completed_epoch']) if previous is not None else 0
    if settings['epochs'] < completed_epoch:
        raise ValueError(f'Target --epochs must be at least completed epoch {completed_epoch}')
    metadata = {}
    summary_path = data_dir/'dataset_summary.json'
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding='utf-8-sig'))
        for key in ('synthetic_only', 'label_semantics'):
            if key in summary:
                metadata[key] = summary[key]
        if 'label_semantics' in metadata:
            metadata['generation_label_semantics'] = metadata['label_semantics']
    metadata['label_mapping_authority'] = 'Current labels.json and split CSV label/label_index; generation-time labels are not training targets'
    if model_family == 'LSTM-VAE':
        metadata.update(lstm_frame_size=config.frame_size, recurrent_steps=config.recurrent_steps,
                        lstm_frame_padding_frames=config.recurrent_steps*config.frame_size-input_length,
                        framing_preserves_all_samples=True)
    output = prepare_run_directory(args.output_dir, model_family, 'classification')
    torch.set_num_threads(settings['threads'])
    device = resolve_device(settings['device'])
    random.seed(settings['seed']); np.random.seed(settings['seed']); torch.manual_seed(settings['seed'])
    model_kwargs = {}
    if context_dim:
        model_kwargs = {'context_dim': context_dim, 'context_embedding_dim': context_embedding_dim}
    model = StateClassifier(config, num_classes=len(class_names), **model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=1e-4)
    generators = {split: torch.Generator().manual_seed(settings['seed']+index) for index, split in enumerate(SPLITS)}
    datasets = {split: RawFourClassDataset(data_dir, subset, normalizer, input_length,
                                           valid_frames=valid_frames, context_vectors=context_vectors)
                for split, subset in subsets.items()}
    loaders = {split: DataLoader(dataset, batch_size=settings['batch_size'], shuffle=(split == 'train'),
                                  generator=generators[split], num_workers=0)
               for split, dataset in datasets.items()}
    if previous is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(previous['optimizer_state_dict'])
        history = deepcopy(previous['history']); evaluations = deepcopy(previous['evaluations'])
        restore_history_metrics(history, evaluations)
        best_metric, best_epoch = previous['best_metric'], previous['best_epoch']
        best_state = deepcopy(previous['best_model_state_dict'])
        total_training_seconds = float(previous.get('training_seconds', 0))
    else:
        history = []; evaluations = []; best_metric = None; best_epoch = None; best_state = None
        total_training_seconds = 0.
    start_epoch = completed_epoch

    def payload(epoch, model_state=None, with_training_state=True):
        value = {'format_version': 1, 'task': 'classification', 'model_family': model_family,
                 'model_config': asdict(config), 'sample_rate': SAMPLE_RATE, 'valid_frames': valid_frames,
                 'input_length': input_length, 'class_names': class_names, 'num_classes': len(class_names),
                 'normalizer': deepcopy(normalizer), 'normalizer_fitted_to': 'train CSV only, complete valid frames',
                 'context_dim': context_dim, 'context_embedding_dim': context_embedding_dim,
                 'context_normalizer': None if context_normalizer is None else context_normalizer.to_dict(),
                 'context_source': settings.get('cnc_context'),
                 'model_state_dict': clone_model_state(model) if model_state is None else model_state,
                 'checkpoint_epoch': epoch, 'scores_calibrated': False, 'device': str(device),
                 'resumed_from': str(resume_path) if resume_path else None, **metadata}
        value['training_state'] = None
        if with_training_state:
            value['training_state'] = {
                'completed_epoch': epoch, 'optimizer_state_dict': optimizer.state_dict(),
                'rng_state': capture_rng_state(generators, device=device), 'history': history, 'evaluations': evaluations,
                'settings': dict(settings), 'data_identity': identity,
                'best_metric': best_metric, 'best_epoch': best_epoch, 'best_model_state_dict': best_state,
                'training_seconds': total_training_seconds,
            }
        return value

    atomic_json({**settings, 'resolved_device': str(device), 'task': 'classification', 'model_family': model_family,
                 'model_config': asdict(config), 'valid_frames': valid_frames, 'input_length': input_length,
                 'padding_frames': input_length-valid_frames, 'sample_rate': SAMPLE_RATE,
                 'class_names': class_names, 'num_classes': len(class_names),
                 'context_dim': context_dim, 'context_embedding_dim': context_embedding_dim,
                 'context_normalizer': None if context_normalizer is None else context_normalizer.to_dict(),
                 'normalizer': normalizer, 'model_selection': 'validation sample macro F1 only',
                 'resumed_from': str(resume_path) if resume_path else None, **metadata}, output/'training_config.json')
    atomic_json({str(index): name for index, name in enumerate(class_names)}, output/'labels.json')
    atomic_json(rows, output/'data_manifest.json')
    if best_state is not None:
        # Retain a complete historical best checkpoint when it is available.
        candidates = [resume_path.parent/'best.pt', resume_path.parent/'checkpoints'/f'epoch_{best_epoch:04d}.pt',
                      resume_path.parent/f'epoch_{best_epoch:04d}.pt']
        inherited = None
        for candidate in candidates:
            if not candidate.is_file():
                continue
            value = torch.load(candidate, map_location='cpu', weights_only=True)
            if (value.get('task') == 'classification' and value.get('model_family') == model_family
                    and value.get('class_names') == class_names and value.get('checkpoint_epoch') == best_epoch
                    and value.get('normalizer') == normalizer
                    and asdict(ModelConfig(**value['model_config'])) == asdict(config)):
                inherited = value; break
        if inherited is None:
            inherited = payload(best_epoch, best_state, with_training_state=False)
            inherited['resume_capability'] = 'inference_only; full resume remains available from last.pt'
        atomic_torch_save(inherited, output/'best.pt')
    if previous is not None:
        restore_rng_state(previous['rng_state'], generators, device=device)
    atomic_torch_save(payload(completed_epoch), output/'last.pt')
    _save_history(output, history, evaluations)
    print(f'Model: {model_family}; Task: classification; Classes: {len(class_names)}', flush=True)
    print(f'Device: {device}', flush=True)
    print(f'Run directory: {output}', flush=True)
    print(f'Dataset: {data_dir}', flush=True)
    if settings['csv'] is not None:
        print(f'CSV: {settings["csv"]}', flush=True)
    print(f'Input: {valid_frames} raw frames at {SAMPLE_RATE} Hz; model length {input_length}; batch {settings["batch_size"]}', flush=True)
    if model_family == 'LSTM-VAE':
        print(f'LSTM framing: {config.frame_size} sample frames per step '
              f'({1000*config.frame_size/SAMPLE_RATE:.3f} ms); recurrent steps: {config.recurrent_steps}', flush=True)
    if previous is not None:
        print(f'Full-state resume after epoch {completed_epoch}.', flush=True)
    try:
        for epoch in range(completed_epoch+1, settings['epochs']+1):
            started = time.perf_counter(); model.train()
            sums = {'loss': 0., 'ce_loss': 0., 'reconstruction_loss': 0., 'kl_loss': 0.}; count = 0
            for batch_index, batch in enumerate(loaders['train'], start=1):
                signal, context, labels = _batch_parts(batch)
                signal, labels = signal.to(device), labels.to(device)
                if context is not None:
                    context = context.to(device)
                optimizer.zero_grad(set_to_none=True)
                kwargs = {} if context is None else {'context': context}
                values = joint_loss(model(signal, reconstruct=settings['reconstruction_weight'] > 0, **kwargs),
                                    labels, signal, valid_frames, settings['reconstruction_weight'], settings['beta'])
                if not torch.isfinite(values['loss']):
                    raise RuntimeError('Non-finite classification training loss')
                values['loss'].backward()
                clip_gradients(model.parameters(), settings['gradient_clip'])
                optimizer.step(); count += len(labels)
                for key in sums:
                    sums[key] += float(values[key].detach())*len(labels)
                if batch_index == 1 or batch_index % 50 == 0:
                    print(f'Epoch {epoch}/{settings["epochs"]}: batch {batch_index}/{len(loaders["train"])}; '
                          f'loss={sums["loss"]/count:.6g}; CE={sums["ce_loss"]/count:.6f}; '
                          f'MSE={sums["reconstruction_loss"]/count:.6g}; KL={sums["kl_loss"]/count:.6g}', flush=True)
            should_evaluate = epoch % settings['eval_every'] == 0 or epoch == settings['epochs']
            val = evaluate(model, loaders['val'], class_names)[0] if should_evaluate else None
            test = evaluate(model, loaders['test'], class_names)[0] if should_evaluate and settings['test_during_training'] else None
            improved = val is not None and (best_metric is None or val['macro_f1'] > best_metric)
            if improved:
                best_metric, best_epoch = val['macro_f1'], epoch
                best_state = clone_model_state(model)
            total_training_seconds += time.perf_counter()-started
            row = {'epoch': epoch, **{f'train_{key}': value/count for key, value in sums.items()},
                   'val_accuracy': val['accuracy'] if val else None,
                   'val_macro_precision': val['macro_precision'] if val else None,
                   'val_macro_recall': val['macro_recall'] if val else None,
                   'val_macro_f1': val['macro_f1'] if val else None,
                   'test_accuracy': test['accuracy'] if test else None,
                   'test_macro_precision': test['macro_precision'] if test else None,
                   'test_macro_recall': test['macro_recall'] if test else None,
                   'test_macro_f1': test['macro_f1'] if test else None,
                   'validation_evaluated': val is not None, 'test_evaluated': test is not None}
            history.append(row)
            if should_evaluate:
                evaluations.append({'epoch': epoch, 'validation': val, 'test': test})
            completed_epoch = epoch
            saved = payload(completed_epoch)
            atomic_torch_save(saved, output/'last.pt')
            if should_evaluate:
                atomic_torch_save(saved, output/'checkpoints'/f'epoch_{epoch:04d}.pt')
            if improved:
                atomic_torch_save(saved, output/'best.pt')
            _save_history(output, history, evaluations)
            message = (f'{model_family}: loss={row["train_loss"]:.6g}; CE={row["train_ce_loss"]:.6f}; '
                       f'MSE={row["train_reconstruction_loss"]:.6g}; KL={row["train_kl_loss"]:.6g}')
            lines = [section_heading(f'Training | Epoch {epoch}/{settings["epochs"]}'), message]
            if val is not None:
                lines.extend([section_heading(f'Validation | Epoch {epoch}/{settings["epochs"]}'),
                              format_classification_metrics(val)])
            if test is not None:
                lines.extend([section_heading(f'Test | Epoch {epoch}/{settings["epochs"]}'),
                              format_classification_metrics(test)])
            lines.append(section_heading(f'End of Epoch {epoch}'))
            print('\n'.join(lines), flush=True)
        final = payload(completed_epoch)
        atomic_torch_save(final, output/'last.pt'); atomic_torch_save(final, output/'final.pt')
        if best_state is None:
            raise RuntimeError('No validation-selected checkpoint exists')
        model.load_state_dict(best_state)
        validation_metrics, _, _ = evaluate(model, loaders['val'], class_names)
        test_metrics, test_truth, test_scores = evaluate(model, loaders['test'], class_names)
        baseline_class = int(train_counts.argmax())
        majority = classification_metrics(test_truth, np.full(len(test_truth), baseline_class), class_names)
        metrics = {'task': 'classification', 'model_family': model_family, 'device': str(device), 'class_names': class_names,
                   'num_classes': len(class_names), 'evaluation_checkpoint': str(output/'best.pt'),
                   'evaluation_weights': 'validation-selected best', 'evaluation_epoch': best_epoch,
                   'best_validation_macro_f1': best_metric, 'completed_epoch': completed_epoch,
                   'epochs_this_invocation': completed_epoch-start_epoch, 'training_seconds': total_training_seconds,
                   'validation': validation_metrics, 'test': test_metrics,
                   'majority_baseline': {'class_id': baseline_class, 'metrics': majority},
                   'sample_counts': {split: len(subset) for split, subset in subsets.items()},
                   'label_counts': {split: np.bincount([row['label_index'] for row in subset], minlength=len(class_names)).tolist()
                                    for split, subset in subsets.items()},
                   'sample_rate': SAMPLE_RATE, 'valid_frames': valid_frames, 'input_length': input_length,
                   'padding_frames': input_length-valid_frames,
                   'scores_calibrated': False, 'validated_for_real_machine_faults': False,
                   'final_checkpoint': str(output/'final.pt'), 'last_checkpoint': str(output/'last.pt'),
                   'model_input': ('Full raw stereo waveform plus normalized CNC context; labels come from the dataset CSV'
                                   if context_dim else
                                   'Full raw stereo waveform only; no label-generation parameters or metadata features'),
                   'test_selection_policy': 'Periodic test is observational only; best uses validation macro F1', **metadata}
        atomic_json(metrics, output/'metrics.json')
        predictions = []
        for source, truth, probabilities in zip(subsets['test'], test_truth, test_scores):
            predicted = int(probabilities.argmax())
            predictions.append({'sample_id': source['sample_id'], 'base_sample_id': source['base_sample_id'],
                'recording_id': source['recording_id'], 'source_recording_id': source['source_recording_id'],
                'wav_file': source['wav_file'], 'absolute_wav_file': str((data_dir/source['wav_file']).resolve()),
                'true_label_index': int(truth), 'true_label': class_names[int(truth)],
                'predicted_label_index': predicted, 'predicted_label': class_names[predicted],
                **{f'probability_class_{index}': float(value) for index, value in enumerate(probabilities)}})
        _write_csv(output/'test_predictions.csv', predictions)
        best_lines = [
            section_heading(f'Best Checkpoint | Epoch {best_epoch}'),
            f'Selected by highest validation Macro F1={best_metric:.6f}',
            section_heading(f'Validation | Best Epoch {best_epoch}'),
            format_classification_metrics(validation_metrics),
            section_heading(f'Test | Best Epoch {best_epoch}'),
            format_classification_metrics(test_metrics),
            section_heading('End of Best Checkpoint Summary'),
        ]
        print('\n'.join(best_lines), flush=True)
        return 0
    except KeyboardInterrupt:
        saved = torch.load(output/'last.pt', map_location='cpu', weights_only=True)
        epoch = saved['training_state']['completed_epoch']
        atomic_json({'status': 'interrupted', 'last_completed_epoch': epoch,
                     'resume_checkpoint': str(output/'last.pt')}, output/'interruption.json')
        print(f'Training interrupted. Last completed epoch: {epoch}. Resume from {output / "last.pt"}', flush=True)
        return 130
