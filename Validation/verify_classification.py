"""Evaluate a saved classifier on test.csv and compare file/chunk predictions.

Run after training has finished. This script never fits preprocessing statistics
or updates model weights; each test WAV starts a separate streaming session.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Dataset.Preprocessing.fourclass_data import load_manifest, load_class_names, read_wav
from fourclass_prediction import WavClassifier
from fourclass_training import classification_metrics, format_classification_metrics
from training_support import atomic_json


def resolve_evaluation_data_dir(checkpoint, data_dir=None):
    """A saved experiment keeps its test dataset even after a new version is active."""
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()
    state = checkpoint.get('training_state') or {}
    saved = state.get('settings', {}).get('data_dir') or state.get('data_identity', {}).get('data_dir')
    if not saved:
        raise ValueError('Checkpoint has no saved dataset directory; provide --data-dir explicitly')
    return Path(saved).expanduser().resolve()


def verify_test_set(engine, data_dir, output_dir, chunk_ms=20.):
    """Use the current CSV targets; check the checkpoint's held-out sources."""
    from classification_streaming import StreamingClassifier

    checkpoint = torch.load(engine.checkpoint_file, map_location='cpu', weights_only=True)
    data_dir = resolve_evaluation_data_dir(checkpoint, data_dir)
    output_dir = Path(output_dir).expanduser().resolve()
    names = list(load_class_names(data_dir))
    if engine.class_names != names:
        raise ValueError('Dataset labels.json differs from checkpoint class names/order')
    saved_csv = (checkpoint.get('training_state') or {}).get('settings', {}).get('csv')
    rows = [row for row in load_manifest(data_dir, csv_path=saved_csv) if row['split'] == 'test']
    saved = (checkpoint.get('training_state') or {}).get('data_identity', {}).get('ordered_samples')
    if saved is None:
        raise ValueError('This test-set audit requires a checkpoint containing its training split identity')
    training_sources = {row['source_recording_id'] for row in saved if row['split'] == 'train'}
    test_sources = {row['source_recording_id'] for row in rows}
    if training_sources & test_sources:
        raise ValueError('Current test.csv shares an original source with checkpoint training data')
    identity_fields = ('sample_id', 'source_recording_id', 'label', 'label_index', 'wav_file')
    current_ids = sorted(tuple(row[key] for key in identity_fields) for row in rows)
    saved_ids = sorted(tuple(row[key] for key in identity_fields) for row in saved if row['split'] == 'test')
    if current_ids != saved_ids:
        raise ValueError('Current test.csv does not match the saved experiment test identities and labels')
    if not np.isfinite(chunk_ms) or chunk_ms <= 0:
        raise ValueError('chunk_ms must be finite and positive')
    chunk_frames = max(1, round(engine.sample_rate * chunk_ms / 1000))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions, true_indices, predicted_indices = [], [], []
    largest_delta = 0.
    all_labels_match = True
    started = time.perf_counter()
    with (output_dir/'stream_predictions.jsonl').open('w', encoding='utf-8') as stream_file:
        for index, row in enumerate(rows):
            path = data_dir/row['wav_file']
            offline = engine.predict_wav(path)
            sample_rate, pcm = read_wav(path, expected_frames=engine.valid_frames)
            stream = StreamingClassifier(engine)
            emitted = []
            for start in range(0, len(pcm), chunk_frames):
                emitted.extend(stream.push(pcm[start:start+chunk_frames], sample_rate=sample_rate))
            tail = stream.finish()
            if len(emitted) != 1 or tail['trailing_frames'] != 0:
                raise RuntimeError(f"{row['sample_id']}: a five-second sample must emit exactly one complete window")
            online = emitted[0]
            offline_scores = np.asarray([offline['class_probabilities'][name] for name in names])
            online_scores = np.asarray([online['class_probabilities'][name] for name in names])
            delta = float(np.max(np.abs(offline_scores-online_scores)))
            largest_delta = max(largest_delta, delta)
            matched = offline['predicted_class_index'] == online['predicted_class_index']
            all_labels_match = all_labels_match and matched
            if not np.allclose(offline_scores, online_scores, atol=1e-6, rtol=1e-5) or not matched:
                raise RuntimeError(f"{row['sample_id']}: file/stream class predictions differ")
            true_indices.append(row['label_index'])
            predicted_indices.append(offline['predicted_class_index'])
            predictions.append({
                'sample_id': row['sample_id'], 'source_recording_id': row['source_recording_id'],
                'recording_id': row['recording_id'], 'wav_file': row['wav_file'],
                'start_frame_in_recording': row['start_frame'],
                'true_label_index': row['label_index'], 'true_label': row['label'],
                'predicted_label_index': offline['predicted_class_index'], 'predicted_label': offline['predicted_label'],
                'confidence': offline['confidence'], 'stream_label_matches': matched,
                'max_stream_probability_difference': delta,
                **{f'probability_class_{i}': float(value) for i, value in enumerate(offline_scores)},
            })
            stream_file.write(json.dumps({'sample_id': row['sample_id'], **online}, ensure_ascii=False, allow_nan=False)+'\n')
            if (index+1) % 50 == 0:
                print(f'Verified file/stream predictions: {index+1}/{len(rows)} samples', flush=True)
    metrics = classification_metrics(true_indices, predicted_indices, names)
    checkpoint_epoch = checkpoint.get('checkpoint_epoch')
    result = {
        'task': 'classification', 'model_family': engine.model_family,
        'device': str(engine.device),
        'checkpoint_file': str(engine.checkpoint_file), 'checkpoint_epoch': checkpoint_epoch,
        'data_dir': str(data_dir), 'split': 'test', 'class_names': names,
        'csv_file': saved_csv,
        'test_source_groups': sorted(test_sources), 'training_source_overlap': [],
        'test_identity_matches_checkpoint': True, 'metrics': metrics,
        'stream_equivalence': {'sample_count': len(rows), 'chunk_frames': chunk_frames,
                               'all_labels_match': all_labels_match,
                               'max_absolute_probability_difference': largest_delta,
                               'partial_tail_frames': 0, 'sessions_reset_per_file': True},
        'normalizer_refitted': False, 'weights_updated': False,
        'wall_time_seconds': time.perf_counter()-started,
        'confidence_uncalibrated': True, **engine.label_metadata,
    }
    if 'good' in names:
        result['original_good_subset'] = metrics['per_class'][names.index('good')]
    with (output_dir/'test_predictions.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader(); writer.writerows(predictions)
    atomic_json({str(i): name for i, name in enumerate(names)}, output_dir/'labels.json')
    atomic_json(result, output_dir/'verification.json')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['cnn', 'lstm'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, help='Default: the dataset saved in this checkpoint; explicit paths override it')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--chunk-ms', type=float, default=20.)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto',
                        help='auto: use CUDA after a successful probe, otherwise CPU (default)')
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    family = {'cnn': 'CNN1D-VAE', 'lstm': 'LSTM-VAE'}[args.model]
    sys.path.insert(0, str(PROJECT_ROOT/family))
    from classifier import StateClassifier
    from model import ModelConfig
    torch.set_num_threads(args.threads)
    engine = WavClassifier(args.checkpoint, family, StateClassifier, ModelConfig, device=args.device)
    print('Checkpoint loaded successfully.', flush=True)
    print(f'Device: {engine.device}', flush=True)
    result = verify_test_set(engine, args.data_dir, args.output_dir, args.chunk_ms)
    print('-'*64)
    print(f"Test {format_classification_metrics(result['metrics'])}")
    print(f"File/stream labels agree: {result['stream_equivalence']['all_labels_match']}")
    print(f"Output: {args.output_dir.resolve()}")
    print('-'*64)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
