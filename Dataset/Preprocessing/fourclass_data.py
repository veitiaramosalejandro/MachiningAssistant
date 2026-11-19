"""Load complete WAV samples using editable class definitions and CSV indexes.

Each sample is the complete original five-second stereo PCM waveform. Signal
preprocessing consists only of training-statistic normalization and optional
trailing normalized-zero padding for the convolutional model.
"""
from __future__ import annotations

import csv
import json
import math
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .active_dataset import LEGACY_DATA_DIR, resolve_dataset_dir


CLASS_NAMES = ('good', 'sufficient', 'bad', 'very_bad')
VALID_FRAMES = 220500
SAMPLE_RATE = 44100
FS = SAMPLE_RATE
CHANNELS = 2
DEFAULT_DATA_DIR = LEGACY_DATA_DIR
SPLITS = ('train', 'val', 'test')
STRING_FIELDS = ('sample_id', 'base_sample_id', 'recording_id', 'source_recording_id', 'split', 'label', 'wav_file')
INT_FIELDS = ('label_index', 'frame_count', 'start_frame', 'end_frame_exclusive')
FLOAT_FIELDS = ('start_seconds', 'end_seconds_exclusive', 'duration_seconds')


def model_input_length(model_family, valid_frames=VALID_FRAMES):
    """Keep all valid samples; CNN adds only the trailing stride-alignment pad."""
    if not isinstance(valid_frames, (int, np.integer)) or valid_frames < 1:
        raise ValueError('valid_frames must be a positive integer')
    if model_family == 'CNN1D-VAE':
        return ((int(valid_frames)+63)//64)*64
    if model_family == 'LSTM-VAE':
        return int(valid_frames)
    raise ValueError('model_family must be CNN1D-VAE or LSTM-VAE')


def load_class_names(data_dir=None):
    """Read consecutive string indices from labels.json; classes are dataset-defined."""
    data_dir = resolve_dataset_dir(data_dir)
    mapping = json.loads((data_dir/'labels.json').read_text(encoding='utf-8-sig'))
    if not isinstance(mapping, dict) or len(mapping) < 2 or set(mapping) != {str(i) for i in range(len(mapping))}:
        raise ValueError('labels.json must map consecutive string indices 0..N-1 with N >= 2')
    names = []
    for index in range(len(mapping)):
        value = mapping[str(index)]
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Class names in labels.json must be nonempty strings')
        names.append(value.strip())
    if len(set(names)) != len(names):
        raise ValueError('Class names in labels.json must be unique')
    return tuple(names)


def _canonical_row(row, valid_frames, class_names, validate_labels=True):
    missing = set((*STRING_FIELDS, *INT_FIELDS, *FLOAT_FIELDS))-set(row)
    if missing:
        raise ValueError(f'Missing sample fields: {sorted(missing)}')
    result = {key: str(row[key]).strip() if row[key] is not None else '' for key in STRING_FIELDS}
    if any(not result[key] for key in STRING_FIELDS):
        raise ValueError('Sample IDs, split, label, and wav_file must be nonempty')
    for key in INT_FIELDS:
        value = row[key]
        if isinstance(value, bool):
            raise ValueError(f'{key} must be an integer')
        result[key] = int(value)
        if str(result[key]) != str(value).strip():
            raise ValueError(f'{key} must be an integer: {value!r}')
    for key in FLOAT_FIELDS:
        result[key] = float(row[key])
        if not math.isfinite(result[key]):
            raise ValueError(f'{key} must be finite')
    index = result['label_index']
    if validate_labels and (not 0 <= index < len(class_names) or result['label'] != class_names[index]):
        raise ValueError('label and label_index do not match labels.json')
    if result['frame_count'] != valid_frames or result['end_frame_exclusive']-result['start_frame'] != valid_frames:
        raise ValueError(f"{result['sample_id']}: expected the complete {valid_frames}-frame sample")
    if result['start_frame'] < 0:
        raise ValueError('start_frame must be nonnegative')
    expected_times = {
        'start_seconds': result['start_frame']/SAMPLE_RATE,
        'end_seconds_exclusive': result['end_frame_exclusive']/SAMPLE_RATE,
        'duration_seconds': valid_frames/SAMPLE_RATE,
    }
    if any(not math.isclose(result[key], value, rel_tol=0, abs_tol=1e-8) for key, value in expected_times.items()):
        raise ValueError(f"{result['sample_id']}: frame and time metadata disagree")
    return result


def _wav_path(data_dir, relative_path):
    relative_path = Path(relative_path)
    if relative_path.is_absolute():
        raise ValueError('wav_file must be relative to data_dir')
    path = (Path(data_dir)/relative_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'WAV sample not found: {path}')
    return path


def load_manifest(data_dir=None, valid_frames=VALID_FRAMES, *, csv_path=None):
    """Read an explicit combined CSV, or the legacy train/val/test CSV files.

    Returns one list in CSV order (train, then val, then test). Amplitude,
    phase, and other generation fields are not returned as model features.
    An explicit CSV groups samples by its split column and skips exclude rows.
    WAV paths always resolve against data_dir, not the CSV's parent directory.
    """
    data_dir = resolve_dataset_dir(data_dir)
    class_names = load_class_names(data_dir)
    frozen = None
    metadata_path = data_dir/'samples.json'
    if metadata_path.is_file():
        original = json.loads(metadata_path.read_text(encoding='utf-8-sig'))
        frozen = {}
        for raw in original:
            if raw.get('split') not in SPLITS:
                continue
            row = _canonical_row(raw, valid_frames, class_names, validate_labels=False)
            if row['sample_id'] in frozen:
                raise ValueError(f"Duplicate sample_id in samples.json: {row['sample_id']}")
            frozen[row['sample_id']] = row
    identity_fields = tuple(key for key in (*STRING_FIELDS, *INT_FIELDS, *FLOAT_FIELDS)
                            if key not in ('label', 'label_index'))
    combined = None
    if csv_path is not None:
        csv_path = Path(csv_path).expanduser().resolve()
        combined = {split: [] for split in SPLITS}
        with csv_path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if 'split' not in (reader.fieldnames or []):
                raise ValueError(f'Combined CSV {csv_path} must contain a split column')
            for raw in reader:
                split = (raw.get('split') or '').strip()
                if split not in (*SPLITS, 'exclude'):
                    raise ValueError(f'Unknown split {split!r} in {csv_path}; use train, val, test or exclude')
                if split == 'exclude':
                    continue
                combined[split].append(_canonical_row(raw, valid_frames, class_names))
        missing_splits = [split for split, selected in combined.items() if not selected]
        if missing_splits:
            raise ValueError(f'Combined CSV {csv_path} must include nonempty train, val and test splits; missing: {", ".join(missing_splits)}')
    rows = []
    seen_ids, seen_paths = set(), set()
    by_base, by_source = defaultdict(list), defaultdict(set)
    for split in SPLITS:
        if combined is None:
            with (data_dir/f'{split}.csv').open(encoding='utf-8-sig', newline='') as handle:
                selected = [_canonical_row(row, valid_frames, class_names) for row in csv.DictReader(handle)]
        else:
            selected = combined[split]
        if not selected:
            raise ValueError(f'{split}.csv must be nonempty')
        if split == 'train' and {row['label_index'] for row in selected} != set(range(len(class_names))):
            raise ValueError('train.csv must contain every configured class from labels.json')
        for row in selected:
            if row['split'] != split:
                raise ValueError(f"{row['sample_id']}: split does not match {split}.csv")
            if row['sample_id'] in seen_ids:
                raise ValueError(f"Duplicate sample_id: {row['sample_id']}")
            if frozen is not None and (row['sample_id'] not in frozen or any(
                row[key] != frozen[row['sample_id']][key] for key in identity_fields
            )):
                raise ValueError(f"{row['sample_id']}: CSV differs from frozen samples.json; rebuild the dataset to change assignments")
            path = _wav_path(data_dir, row['wav_file'])
            if path in seen_paths:
                raise ValueError(f'Duplicate physical WAV path: {path}')
            seen_ids.add(row['sample_id']); seen_paths.add(path)
            by_base[row['base_sample_id']].append(row)
            by_source[row['source_recording_id']].add(split)
            rows.append(row)
    if frozen is not None and seen_ids != set(frozen):
        raise ValueError('CSV membership differs from frozen samples.json; rebuild the dataset to change assignments')
    if any(len(splits) != 1 for splits in by_source.values()):
        raise ValueError('A source_recording_id appears across train/val/test splits')
    for base, variants in by_base.items():
        identity = {(row['split'], row['source_recording_id']) for row in variants}
        if len(identity) != 1:
            raise ValueError(f'{base}: base samples cross splits or original sources')
    return rows


def read_wav(path, expected_frames=None):
    """Read a stereo PCM16 WAV as float32 raw counts without amplitude scaling."""
    path = Path(path)
    with wave.open(str(path), 'rb') as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) != (SAMPLE_RATE, CHANNELS, 2, 'NONE'):
            raise ValueError(f'{path}: expected 44100 Hz stereo PCM16 WAV')
        frames = wav.getnframes()
        if expected_frames is not None and frames != expected_frames:
            raise ValueError(f'{path}: expected exactly {expected_frames} frames, found {frames}; no cropping or resampling is performed')
        payload = wav.readframes(frames)
    if len(payload) != frames*CHANNELS*2:
        raise ValueError(f'{path}: incomplete PCM payload')
    return SAMPLE_RATE, np.frombuffer(payload, dtype='<i2').reshape(frames, CHANNELS).astype(np.float32)


def fit_normalizer(data_dir, train_rows):
    """Fit population channel statistics using the full training WAVs only."""
    train_rows = list(train_rows)
    if not train_rows or any(row['split'] != 'train' for row in train_rows):
        raise ValueError('Normalizer fitting accepts nonempty training rows only')
    total = np.zeros(CHANNELS, dtype=np.float64)
    squared = np.zeros(CHANNELS, dtype=np.float64)
    count = 0
    for row in train_rows:
        _, pcm = read_wav(_wav_path(data_dir, row['wav_file']), expected_frames=row['frame_count'])
        values = pcm.astype(np.float64)
        total += values.sum(axis=0, dtype=np.float64)
        squared += np.square(values).sum(axis=0, dtype=np.float64)
        count += len(values)
    mean = total/count
    scale = np.maximum(np.sqrt(np.maximum(squared/count-mean*mean, 0.)), 1.)
    return {'mean_pcm': mean.tolist(), 'scale_pcm': scale.tolist()}


def prepare_signal(pcm, normalizer, input_length, valid_frames=VALID_FRAMES):
    """Normalize every valid frame, then append normalized zero for CNN alignment."""
    allowed = {
        model_input_length('LSTM-VAE', valid_frames),
        model_input_length('CNN1D-VAE', valid_frames)
    }
    if input_length not in allowed:
        raise ValueError('input_length must equal valid_frames or its next multiple of 64')
    values = np.asarray(pcm)
    if values.shape != (valid_frames, CHANNELS):
        raise ValueError(f'Expected the complete ({valid_frames}, 2) signal; no short-window selection is performed')
    if not np.isfinite(values).all():
        raise ValueError('Signal contains non-finite values')
    mean = np.asarray(normalizer['mean_pcm'], dtype=np.float64)
    scale = np.asarray(normalizer['scale_pcm'], dtype=np.float64)
    if mean.shape != (2,) or scale.shape != (2,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Normalizer must have two finite means and positive scales')
    normalized = ((values.astype(np.float64)-mean)/scale).astype(np.float32)
    if input_length == valid_frames:
        return np.ascontiguousarray(normalized)
    output = np.zeros((input_length, CHANNELS), dtype=np.float32)
    output[:valid_frames] = normalized
    return output


class RawFourClassDataset(Dataset):
    """Lazy WAV samples with optional normalized CNC vectors.

    Without ``context_vectors`` the legacy item is ``(signal, label)``. With
    it, the multimodal item is ``(signal, context, label)``.
    """

    def __init__(self, data_dir, rows, normalizer, input_length, valid_frames=VALID_FRAMES,
                 context_vectors=None):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.rows = list(rows)
        self.normalizer = normalizer
        self.input_length = input_length
        self.valid_frames = valid_frames
        self.context_vectors = context_vectors
        self.class_names = load_class_names(self.data_dir)
        if any(not 0 <= row['label_index'] < len(self.class_names) or
               row['label'] != self.class_names[row['label_index']] for row in self.rows):
            raise ValueError('Dataset row labels do not match labels.json')
        if input_length not in {model_input_length('CNN1D-VAE', valid_frames), model_input_length('LSTM-VAE', valid_frames)}:
            raise ValueError('input_length must preserve the complete valid signal with only stride-alignment padding')
        if context_vectors is not None:
            missing = [row['sample_id'] for row in self.rows if row['sample_id'] not in context_vectors]
            if missing:
                raise ValueError(f'Missing CNC context for sample_id: {missing[0]}')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        _, pcm = read_wav(_wav_path(self.data_dir, row['wav_file']), expected_frames=self.valid_frames)
        signal = prepare_signal(pcm, self.normalizer, self.input_length, self.valid_frames)
        label = torch.tensor(row['label_index'], dtype=torch.long)
        if self.context_vectors is None:
            return torch.from_numpy(signal), label
        context = np.asarray(self.context_vectors[row['sample_id']], dtype=np.float32)
        return torch.from_numpy(signal), torch.from_numpy(np.ascontiguousarray(context)), label
