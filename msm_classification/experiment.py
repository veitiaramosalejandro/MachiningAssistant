"""Shared, recording-disjoint training and evaluation for the two MSM classifiers."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Dataset.Preprocessing.dataset_splits import SPLIT_NAMES, read_split_csv, split_recording_ids, write_split_csv
from device_support import resolve_device
from training_support import (prepare_run_directory, atomic_torch_save, atomic_json,
                              capture_rng_state, restore_rng_state, clone_model_state)


CLASS_NAMES=['normal_cutting','abnormal_cutting','defective_tool_normal_cutting']


def resolve_training_split(metadata, split_csv):
    """Read this run's CSV, selecting cached windows without re-preprocessing."""
    rows = read_split_csv(split_csv)
    known = {row['recording_id'] for row in metadata}
    listed = {row['recording_id'] for row in rows}
    unknown = sorted(listed - known)
    if unknown:
        raise ValueError('Split CSV recordings are absent from prepared metadata: ' + ', '.join(unknown))
    if any(row['source_recording_id'] != row['recording_id'] for row in rows):
        raise ValueError('MSM source_recording_id must equal recording_id; source groups cannot be invented')
    split = split_recording_ids(rows)
    ids = {name: set(split[name]) for name in SPLIT_NAMES}
    indices = {name: np.asarray([i for i, row in enumerate(metadata)
                                if row['recording_id'] in ids[name]], dtype=np.int64)
               for name in SPLIT_NAMES}
    selected = set().union(*ids.values())
    provenance = {
        'split_csv_source': str(Path(split_csv).resolve()),
        'split_source_recordings': split,
        'ignored_recordings': sorted(known - selected),
        'excluded_recordings': sorted(row['recording_id'] for row in rows if row['split'] == 'exclude'),
        'unlisted_recordings': sorted(known - listed),
        'ignored_window_count': len(metadata) - sum(len(index) for index in indices.values()),
    }
    return rows, split, indices, provenance


def save_json(path, value):
    atomic_json(value, path)


def active_data_identity(x, y, metadata, indices):
    """Preserve ordered prepared-window identities, without hashing raw signals."""
    return {
        'window_shape': list(x.shape[1:]), 'input_dtype': str(x.dtype),
        'splits': {split: [{'prepared_index': int(i), 'metadata': metadata[int(i)], 'label': int(y[i])}
                           for i in index] for split, index in indices.items()},
    }


def metrics(y, pred):
    precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=[0,1,2],zero_division=0)
    return {'accuracy':float(accuracy_score(y,pred)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1,2]).tolist(),
            'per_class':[{ 'class_id':i,'class_name':CLASS_NAMES[i],'precision':float(precision[i]),
                          'recall':float(recall[i]),'f1':float(f1[i]),'support':int(support[i])} for i in range(3)]}


def grouped_predictions(metadata, y, probabilities):
    groups=defaultdict(list)
    for i,m in enumerate(metadata):groups[m['interval_id']].append(i)
    rows=[]
    for interval_id,indices in groups.items():
        labels=np.unique(y[indices])
        if len(labels)!=1:raise ValueError('Mixed labels inside a labelled cutting interval')
        prob=probabilities[indices].mean(axis=0)
        m=metadata[indices[0]]
        rows.append({'interval_id':interval_id,'recording_id':m['recording_id'],
                     'true_label':int(labels[0]),'predicted_label':int(prob.argmax()),
                     'window_count':len(indices),'class_scores':prob.tolist()})
    return rows


class Windows(Dataset):
    def __init__(self,x,y,indices,mean,scale):
        self.x=x;self.y=y;self.indices=np.asarray(indices);self.mean=mean;self.scale=scale
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        j=self.indices[i]
        z=(np.asarray(self.x[j],dtype=np.float32)-self.mean)/self.scale
        return torch.from_numpy(np.ascontiguousarray(z)),torch.tensor(int(self.y[j]),dtype=torch.long)


def train_normalizer(x,indices):
    count=0;total=np.zeros(2,np.float64);squares=np.zeros(2,np.float64)
    for start in range(0,len(indices),128):
        block=np.asarray(x[indices[start:start+128]],dtype=np.float64)
        count+=block.shape[0]*block.shape[1]
        total+=block.sum(axis=(0,1));squares+=(block*block).sum(axis=(0,1))
    mean=total/count
    scale=np.sqrt(np.maximum(squares/count-mean*mean,0))
    if np.any(scale<=0):raise ValueError('A training input channel is constant')
    return mean.astype(np.float32),scale.astype(np.float32)


def evaluate(model,loader,metadata):
    model.eval();ys=[];probs=[]
    device=next(model.parameters()).device
    with torch.inference_mode():
        for x,y in loader:
            logits=model(x.to(device),reconstruct=False)['logits']
            probabilities=logits.softmax(dim=-1)
            if not torch.isfinite(probabilities).all():raise RuntimeError('Non-finite classification output')
            ys.append(y.cpu().numpy());probs.append(probabilities.cpu().numpy())
    y=np.concatenate(ys);p=np.concatenate(probs)
    intervals=grouped_predictions(metadata,y,p)
    interval_y=np.array([r['true_label'] for r in intervals]);interval_pred=np.array([r['predicted_label'] for r in intervals])
    return {'window_metrics':metrics(y,p.argmax(axis=1)),
            'interval_metrics':metrics(interval_y,interval_pred),
            'interval_predictions':intervals,'y':y,'probabilities':p}


def run_training(StateClassifier, ModelConfig, classification_loss, model_family):
    project = Path(__file__).resolve().parent.parent
    defaults = {
        'prepared': str(project/'Dataset'/'Data'/'msm_prepared'),
        'split_csv': str(project/'Dataset'/'Preprocessing'/'msm_dataset_splits.csv'),
        'epochs': 30, 'patience': 7, 'batch_size': 32, 'lr': .001,
        'reconstruction_weight': .05, 'beta': .001, 'threads': 2, 'seed': 42,
        'eval_every': 20, 'test_during_training': True,
        'device': 'auto',
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--split-csv', type=Path)
    parser.add_argument('--output', type=Path, help='New empty run directory; omitted: timestamped runs/train directory')
    parser.add_argument('--resume', type=Path, help='Resume a full checkpoint, or continue legacy classifier weights')
    parser.add_argument('--epochs', type=int, help='Total target epoch, including epochs already completed')
    parser.add_argument('--patience', type=int, help='Early-stopping patience in validation evaluation checks (default 7)')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--reconstruction-weight', type=float)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--threads', type=int)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--eval-every', type=int, help='Evaluate/save periodic checkpoint every N epochs (default 20); also final epoch')
    parser.add_argument('--test-during-training', action=argparse.BooleanOptionalAction, default=None,
                        help='Observe test metrics at evaluation points (default enabled); never used for selection')
    args = parser.parse_args()
    resume_path = args.resume.expanduser().resolve() if args.resume is not None else None
    resumed = torch.load(resume_path, map_location='cpu', weights_only=True) if resume_path else None
    if resumed is not None:
        if resumed.get('model_family') != model_family or resumed.get('class_names') != CLASS_NAMES:
            raise ValueError('Resume checkpoint is not a matching MSM classifier')
    saved_training = resumed.get('training_state') if resumed else None
    resume_kind = 'full_state' if saved_training else ('weights_only' if resumed else 'fresh')
    saved_settings = saved_training['settings'] if saved_training else {}
    settings = {}
    for name, default in defaults.items():
        supplied = getattr(args, name)
        value = supplied if supplied is not None else saved_settings.get(name, default)
        settings[name] = str(Path(value).expanduser().resolve()) if name in ('prepared', 'split_csv') else value
    locked = ('patience', 'batch_size', 'lr', 'reconstruction_weight', 'beta', 'threads', 'seed')
    if saved_training:
        changed = [name for name in locked if settings[name] != saved_settings[name]]
        if changed:
            raise ValueError('Full resume cannot change optimization settings: ' + ', '.join(changed))
    for name in ('epochs', 'patience', 'batch_size', 'threads', 'eval_every'):
        if settings[name] < 1:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    for name in ('lr', 'reconstruction_weight', 'beta'):
        value = settings[name]
        if not math.isfinite(value) or value < 0 or (name == 'lr' and value == 0):
            parser.error(f'Invalid --{name.replace("_", "-")}')
    torch.set_num_threads(settings['threads'])
    device = resolve_device(settings['device'])
    print(f'Device: {device}', flush=True)
    torch.manual_seed(settings['seed']); np.random.seed(settings['seed']); random.seed(settings['seed'])
    prepared = Path(settings['prepared'])
    x = np.load(prepared/'windows.npy', mmap_mode='r')
    y = np.load(prepared/'labels.npy')
    metadata = json.loads((prepared/'windows.json').read_text())
    preprocessing = json.loads((prepared/'preprocessing.json').read_text())
    if len(x) != len(y) or len(x) != len(metadata):
        raise ValueError('Prepared array/metadata lengths differ')
    split_rows, split, indices, provenance = resolve_training_split(metadata, settings['split_csv'])
    ids = {name: set(split[name]) for name in SPLIT_NAMES}
    for name, index in indices.items():
        if set(y[index].tolist()) != {0, 1, 2}:
            raise ValueError(f'{name} does not contain all three classes')
    identity = active_data_identity(x, y, metadata, indices)
    if resumed:
        if {name: sorted(resumed['split'][name]) for name in SPLIT_NAMES} != {name: sorted(split[name]) for name in SPLIT_NAMES}:
            raise ValueError('Resume CSV changes the active source-recording split')
        if resumed['preprocessing'] != preprocessing:
            raise ValueError('Resume preprocessing configuration differs')
        if saved_training and saved_training['data_identity'] != identity:
            raise ValueError('Resume ordered prepared-window identity, shape, or labels differ')
        mean = np.asarray(resumed['normalizer']['mean'], dtype=np.float32)
        scale = np.asarray(resumed['normalizer']['scale'], dtype=np.float32)
        cfg = ModelConfig(**resumed['model_config'])
        if (cfg.sequence_length, cfg.input_channels) != tuple(x.shape[1:]):
            raise ValueError('Resume model input shape differs from prepared windows')
    else:
        mean, scale = train_normalizer(x, indices['train'])
        cfg = ModelConfig(sequence_length=x.shape[1], input_channels=x.shape[2], latent_dim=16)
    normalizer = {'mean': mean.tolist(), 'scale': scale.tolist()}
    model = StateClassifier(cfg).to(device)
    if resumed:
        model.load_state_dict(resumed['state_dict'])
    datasets = {name: Windows(x, y, index, mean, scale) for name, index in indices.items()}
    generators = {name: torch.Generator().manual_seed(settings['seed']) for name in SPLIT_NAMES}
    loaders = {name: DataLoader(dataset, batch_size=settings['batch_size'], shuffle=(name == 'train'),
                               num_workers=0, generator=generators[name]) for name, dataset in datasets.items()}
    eval_metadata = {name: [metadata[int(i)] for i in indices[name]] for name in ('val', 'test')}
    train_counts = np.bincount(y[indices['train']], minlength=3)
    weights = torch.tensor(len(indices['train'])/(3*train_counts), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=1e-4)
    history = []; best_score = -1.; best_epoch = 0; best_state = None; best_context = None
    stale = 0; completed_epoch = 0; completed_seconds = 0.; resume_events = []
    if saved_training:
        completed_epoch = int(saved_training['epoch'])
        history = copy.deepcopy(saved_training['history'])
        best_score = float(saved_training['best_score']); best_epoch = int(saved_training['best_epoch'])
        best_state = copy.deepcopy(saved_training['best_state_dict'])
        best_context = copy.deepcopy(saved_training['best_context'])
        stale = int(saved_training['stale_evaluations'])
        completed_seconds = float(saved_training.get('training_seconds_completed', 0.))
        resume_events = copy.deepcopy(saved_training.get('resume_events', []))
        optimizer.load_state_dict(saved_training['optimizer_state_dict'])
    elif resumed:
        if 'best_epoch' not in resumed:
            raise ValueError('Legacy classifier lacks best_epoch; cannot determine continuation epoch')
        completed_epoch = int(resumed['best_epoch'])
        print('Weights-only continuation: optimizer and RNG states were not saved; exact resume is unavailable.', flush=True)
    if settings['epochs'] < completed_epoch:
        parser.error(f'--epochs is a total target and cannot be below completed epoch {completed_epoch}')
    if resumed:
        changes = {name: {'previous': saved_settings.get(name), 'current': settings[name]}
                   for name in settings if saved_settings.get(name) != settings[name]}
        resume_events.append({'checkpoint': str(resume_path), 'kind': resume_kind,
                              'from_epoch': completed_epoch, 'settings_changes': changes})
    output = prepare_run_directory(args.output, model_family, 'msm')
    start_epoch = completed_epoch
    # Restore after model, optimizer and DataLoader construction/loading.
    if saved_training:
        restore_rng_state(saved_training['rng_state'], generators, device=device)
    if resume_kind == 'weights_only':
        anchor = evaluate(model, loaders['val'], eval_metadata['val'])
        best_score = anchor['interval_metrics']['macro_f1']; best_epoch = completed_epoch
        best_state = clone_model_state(model)
        history.append({'epoch': completed_epoch, 'resumed_anchor': True, 'train_loss': None,
                        'train_ce_loss': None, 'val_interval_macro_f1': best_score,
                        'val_interval_accuracy': anchor['interval_metrics']['accuracy'],
                        'val_window_macro_f1': anchor['window_metrics']['macro_f1'],
                        'test_interval_macro_f1': None, 'test_interval_accuracy': None,
                        'test_window_macro_f1': None, 'seconds': 0., 'evaluation_reason': 'weights_only_resume_anchor'})

    def context():
        return {'epoch': completed_epoch, 'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                'rng_state': capture_rng_state(generators, device=device), 'history': copy.deepcopy(history),
                'settings': dict(settings), 'split': split, 'normalizer': normalizer,
                'preprocessing': preprocessing, 'data_identity': identity,
                'stale_evaluations': stale, 'training_seconds_completed': completed_seconds,
                'resume_events': copy.deepcopy(resume_events)}

    def checkpoint(state_dict, training_context):
        return {'model_family': model_family, 'task': 'msm', 'model_config': asdict(cfg),
                'state_dict': state_dict, 'normalizer': normalizer, 'preprocessing': preprocessing,
                'class_names': CLASS_NAMES, 'best_epoch': best_epoch,
                'validation_interval_macro_f1': best_score, 'status': 'msm_supervised_experimental',
                'split': split, **provenance,
                'training_state': {**training_context, 'best_score': best_score, 'best_epoch': best_epoch,
                                   'best_state_dict': best_state, 'best_context': best_context}}

    initial_context = context()
    if resume_kind == 'weights_only':
        best_context = initial_context
    atomic_torch_save(checkpoint(clone_model_state(model), initial_context), output/'last.pt')
    if best_state is not None:
        restored_best = checkpoint(best_state, best_context)
        atomic_torch_save(restored_best, output/'best.pt')
        atomic_torch_save(restored_best, output/'best_classifier.pt')
    save_json(output/'history.json', history)
    save_json(output/'data_split.json', split)
    write_split_csv(output/'dataset_splits_used.csv', split_rows)
    save_json(output/'training_config.json', {
        'model_family': model_family, 'model_config': asdict(cfg), 'settings': settings,
        'device': str(device),
        'epochs_limit': settings['epochs'], 'patience': settings['patience'], 'patience_unit': 'validation_evaluation_checks',
        'batch_size': settings['batch_size'], 'learning_rate': settings['lr'],
        'reconstruction_weight': settings['reconstruction_weight'], 'beta': settings['beta'], 'seed': settings['seed'],
        'eval_every': settings['eval_every'], 'test_during_training': settings['test_during_training'],
        'class_weights': weights.tolist(), 'preprocessing': preprocessing, **provenance,
        'model_selection': 'Validation interval macro F1 only; periodic test metrics are observation-only; final report evaluates best weights.',
        'initialization': resume_kind, 'resume_from': str(resume_path) if resume_path else None,
        'start_epoch': start_epoch, 'resume_events': resume_events,
    })
    early_stopped = False
    try:
        for epoch in range(start_epoch + 1, settings['epochs'] + 1):
            started = time.perf_counter(); model.train(); total = 0; losses = defaultdict(float)
            for xb, yb in loaders['train']:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(xb, reconstruct=settings['reconstruction_weight'] > 0)
                terms = classification_loss(prediction, yb, xb,
                    reconstruction_weight=settings['reconstruction_weight'], beta=settings['beta'], class_weights=weights)
                if not torch.isfinite(terms['loss']):
                    raise RuntimeError('Non-finite loss')
                terms['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step(); total += len(xb)
                for key, value in terms.items():
                    losses[key] += float(value.detach()) * len(xb)
            evaluate_now = epoch % settings['eval_every'] == 0 or epoch == settings['epochs']
            row = {'epoch': epoch, **{f'train_{key}': value/total for key, value in losses.items()},
                   'val_interval_macro_f1': None, 'val_interval_accuracy': None, 'val_window_macro_f1': None,
                   'test_interval_macro_f1': None, 'test_interval_accuracy': None, 'test_window_macro_f1': None,
                   'evaluation_reason': 'periodic_or_final' if evaluate_now else None}
            improved = False
            if evaluate_now:
                validation = evaluate(model, loaders['val'], eval_metadata['val'])
                score = validation['interval_metrics']['macro_f1']
                row.update(val_interval_macro_f1=score,
                           val_interval_accuracy=validation['interval_metrics']['accuracy'],
                           val_window_macro_f1=validation['window_metrics']['macro_f1'])
                if score > best_score + 1e-6:
                    best_score = score; best_epoch = epoch; stale = 0
                    best_state = clone_model_state(model); improved = True
                else:
                    stale += 1
                if settings['test_during_training']:
                    observed_test = evaluate(model, loaders['test'], eval_metadata['test'])
                    row.update(test_interval_macro_f1=observed_test['interval_metrics']['macro_f1'],
                               test_interval_accuracy=observed_test['interval_metrics']['accuracy'],
                               test_window_macro_f1=observed_test['window_metrics']['macro_f1'])
            row['seconds'] = time.perf_counter() - started
            completed_seconds += row['seconds']
            history.append(row); completed_epoch = epoch
            current_context = context()
            if improved:
                best_context = current_context
            current = checkpoint(clone_model_state(model), current_context)
            atomic_torch_save(current, output/'last.pt')
            save_json(output/'history.json', history)
            if evaluate_now:
                atomic_torch_save(current, output/'checkpoints'/f'epoch_{epoch:04d}.pt')
            if improved:
                atomic_torch_save(current, output/'best.pt')
                atomic_torch_save(current, output/'best_classifier.pt')
            print(json.dumps({'model': model_family, **row}), flush=True)
            if evaluate_now and stale >= settings['patience']:
                early_stopped = True
                break
    except KeyboardInterrupt:
        saved_epoch = int(torch.load(output/'last.pt', map_location='cpu', weights_only=True)['training_state']['epoch'])
        print(f'Interrupted. Last complete epoch {saved_epoch} remains saved in {output / "last.pt"}.', flush=True)
        raise SystemExit(130) from None
    final_checkpoint = checkpoint(clone_model_state(model), context())
    atomic_torch_save(final_checkpoint, output/'final.pt')
    if best_state is None:
        raise RuntimeError('No validated checkpoint is available; extend the total target epoch')
    model.load_state_dict(best_state); model.eval()
    validation = evaluate(model, loaders['val'], eval_metadata['val'])
    test = evaluate(model, loaders['test'], eval_metadata['test'])
    baseline_class = int(np.argmax(train_counts))
    baseline = metrics([row['true_label'] for row in test['interval_predictions']],
                       [baseline_class]*len(test['interval_predictions']))
    report = {'model_family': model_family, 'best_epoch': best_epoch, 'epochs_run': completed_epoch,
        'history_rows': len(history), 'start_epoch': start_epoch,
        'epochs_executed_this_run': completed_epoch-start_epoch, 'early_stopped': early_stopped,
        'training_seconds': completed_seconds, 'parameter_count': sum(p.numel() for p in model.parameters()),
        'class_names': CLASS_NAMES, 'train_class_counts_windows': train_counts.tolist(),
        'split_recording_counts': {name: len(value) for name, value in ids.items()},
        'split_window_counts': {name: len(value) for name, value in indices.items()},
        'split_window_class_counts': {name: np.bincount(y[index], minlength=3).tolist() for name, index in indices.items()},
        **provenance, 'eval_every': settings['eval_every'], 'test_during_training': settings['test_during_training'],
        'patience_unit': 'validation_evaluation_checks', 'resume_kind': resume_kind, 'resume_events': resume_events,
        'validation': {key: validation[key] for key in ('window_metrics', 'interval_metrics')},
        'test': {key: test[key] for key in ('window_metrics', 'interval_metrics')},
        'majority_baseline_interval_metrics': baseline, 'majority_baseline_class': baseline_class,
        'evaluation_unit': 'labelled cutting interval; average class scores across its selected windows',
        'limitation': 'Recording-disjoint same-machine experiment; observed test metrics do not select checkpoints or early stopping.',
        'torch_version': str(torch.__version__), 'device': str(device), 'cpu_threads': settings['threads']}
    save_json(output/'classification_report.json', report)
    save_json(output/'test_interval_predictions.json', test['interval_predictions'])
    save_json(output/'validation_interval_predictions.json', validation['interval_predictions'])
    rows = []
    for metadata_row, label, probability in zip(eval_metadata['test'], test['y'], test['probabilities']):
        rows.append({**metadata_row, 'true_label': int(label), 'predicted_label': int(probability.argmax()),
                     **{f'score_class_{i}': float(probability[i]) for i in range(3)}})
    with (output/'test_window_predictions.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    restored = StateClassifier(ModelConfig(**asdict(cfg))).to(device)
    restored.load_state_dict(best_state); restored.eval()
    first_x, _ = datasets['test'][0]
    with torch.inference_mode():
        first = restored(first_x[None].to(device), reconstruct=False)['logits'].softmax(-1)[0].cpu().numpy()
    if not np.allclose(first, test['probabilities'][0], atol=1e-5, rtol=1e-5):
        raise AssertionError('Saved-classifier replay mismatch')
    save_json(output/'example_state_prediction.json', {'source': eval_metadata['test'][0],
        'true_label': int(test['y'][0]), 'predicted_label': int(first.argmax()),
        'predicted_state': CLASS_NAMES[int(first.argmax())], 'class_scores': first.tolist(),
        'scores_calibrated': False, 'status': 'msm_supervised_experimental'})
    print(json.dumps({'model': model_family, 'best_epoch': best_epoch,
                      'test_interval_metrics': test['interval_metrics'], 'output': str(output)}, ensure_ascii=False), flush=True)
    return 0
