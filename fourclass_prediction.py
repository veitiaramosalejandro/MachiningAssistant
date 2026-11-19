"""Single complete five-second WAV classification using a saved class mapping.

The module name is retained for the initial four-class workflow. Class names and
head size come from the checkpoint, never from the current dataset labels.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from Dataset.Preprocessing.cnc_features import CncFeatureNormalizer, FEATURE_NAMES
from Dataset.Preprocessing import fourclass_data
from device_support import resolve_device


class WavClassifier:
    def __init__(self, checkpoint, model_family, classifier_type, config_type, device='auto'):
        self.checkpoint_file = Path(checkpoint).expanduser().resolve()
        state = torch.load(self.checkpoint_file, weights_only=True, map_location='cpu')
        if state.get('task') not in ('classification', 'fourclass'):
            raise ValueError(
                'Expected a WAV classification checkpoint; reconstruction and MSM checkpoints are unsupported')
        if state.get('model_family') != model_family:
            raise ValueError(f'Checkpoint model_family must be {model_family}')
        if state.get('sample_rate') != fourclass_data.SAMPLE_RATE or state.get(
                'valid_frames') != fourclass_data.VALID_FRAMES:
            raise ValueError('Checkpoint must describe a complete 44100 Hz, 220500-frame WAV sample')
        names = state.get('class_names')
        if (not isinstance(names, (list, tuple)) or len(names) < 2
                or any(not isinstance(name, str) or not name.strip() for name in names)
                or len(set(names)) != len(names)):
            raise ValueError('Checkpoint class_names must contain at least two distinct nonempty names')
        if state.get('num_classes', len(names)) != len(names):
            raise ValueError('Checkpoint num_classes differs from class_names')
        self.class_names = list(names)
        self.sample_rate = state['sample_rate']
        self.valid_frames = state['valid_frames']
        self.input_length = fourclass_data.model_input_length(model_family)
        self.context_dim = int(state.get('context_dim', 0))
        self.context_embedding_dim = int(state.get('context_embedding_dim', 8))
        self.context_normalizer = None
        if self.context_dim:
            if model_family != 'CNN1D-VAE':
                raise ValueError('CNC multimodal inference is supported only for CNN1D-VAE')
            if self.context_dim != len(FEATURE_NAMES):
                raise ValueError('Checkpoint context_dim does not match the CNC feature schema')
            try:
                self.context_normalizer = CncFeatureNormalizer.from_dict(
                    state['context_normalizer']
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('Checkpoint lacks a valid CNC context normalizer') from exc
        config = state.get('model_config')
        if (not isinstance(config, dict) or config.get('sequence_length') != self.input_length
                or config.get('input_channels') != fourclass_data.CHANNELS):
            raise ValueError(f'Checkpoint input shape must be ({self.input_length}, 2) for {model_family}')
        normalizer = state.get('normalizer', {})
        try:
            mean = np.asarray(normalizer['mean_pcm'], dtype=np.float64)
            scale = np.asarray(normalizer['scale_pcm'], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Checkpoint lacks a valid saved PCM normalizer') from exc
        if (mean.shape != (2,) or scale.shape != (2,) or not np.isfinite(mean).all()
                or not np.isfinite(scale).all() or np.any(scale <= 0)):
            raise ValueError('Checkpoint normalizer needs two finite means and positive scales')
        self.normalizer = {'mean_pcm': mean.tolist(), 'scale_pcm': scale.tolist()}
        if 'model_state_dict' not in state:
            raise ValueError('Checkpoint lacks model_state_dict')
        self.device = resolve_device(device)
        model_kwargs = {}
        if self.context_dim:
            model_kwargs = {
                'context_dim': self.context_dim,
                'context_embedding_dim': self.context_embedding_dim,
            }
        self.model = classifier_type(
            config_type(**config), num_classes=len(self.class_names), **model_kwargs
        )
        self.model.load_state_dict(state['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        self.model_family = model_family
        self.framing_metadata = {}
        if model_family == 'LSTM-VAE':
            frame_size = config.get('frame_size', 1)
            recurrent_steps = (self.input_length + frame_size - 1) // frame_size
            self.framing_metadata = {
                'lstm_frame_size': frame_size,
                'recurrent_steps': recurrent_steps,
                'lstm_frame_padding_frames': recurrent_steps * frame_size - self.input_length
            }
        self.label_metadata = {key: state[key] for key in ('label_semantics', 'synthetic_only') if key in state}

    @staticmethod
    def load_context(path):
        """Load one raw MachiningContext JSON object for inference."""
        context_path = Path(path).expanduser().resolve()
        if not context_path.is_file():
            raise FileNotFoundError(f'CNC context JSON not found: {context_path}')
        payload = json.loads(context_path.read_text(encoding='utf-8-sig'))
        if not isinstance(payload, dict):
            raise ValueError('CNC context JSON must contain one object')
        context = payload.get('context', payload)
        # Playback metadata may wrap the actual context one level deeper.
        if isinstance(context, dict) and isinstance(context.get('context'), dict):
            context = context['context']
        if not isinstance(context, dict):
            raise ValueError('CNC context JSON must contain one object')
        return context

    def _prepare_context(self, context):
        if self.context_dim == 0:
            if context is not None:
                raise ValueError('This audio-only checkpoint does not accept CNC context')
            return None
        if context is None:
            raise ValueError('CNC context is required by this multimodal checkpoint')
        if isinstance(context, dict):
            values = self.context_normalizer.transform_context(context)
        else:
            values = np.asarray(context, dtype=np.float32)
            if values.shape != (self.context_dim,) or not np.isfinite(values).all():
                raise ValueError(
                    f'Normalized context must have shape ({self.context_dim},) and finite values'
                )
        return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))[None].to(self.device)

    def predict_wav(self, path, context=None):
        wav_file = Path(path).expanduser().resolve()
        sample_rate, pcm = fourclass_data.read_wav(wav_file, expected_frames=self.valid_frames)
        return {
            **self.predict_pcm(pcm, sample_rate=sample_rate, context=context),
            'wav_file': str(wav_file),
        }

    def predict_pcm(self, pcm, sample_rate=44100, context=None):
        """Classify one complete window of raw PCM counts from a file or stream."""
        if sample_rate != self.sample_rate:
            raise ValueError(f'Expected sample rate {self.sample_rate}, received {sample_rate}')
        prepared = fourclass_data.prepare_signal(
            pcm,
            self.normalizer,
            self.input_length,
            valid_frames=self.valid_frames,
        )
        tensor = torch.from_numpy(np.ascontiguousarray(prepared))[None].to(self.device)
        context_tensor = self._prepare_context(context)
        model_kwargs = {} if context_tensor is None else {'context': context_tensor}
        with torch.inference_mode():
            logits = self.model(tensor, reconstruct=False, **model_kwargs)['logits']
            if tuple(logits.shape) != (1, len(self.class_names)) or not torch.isfinite(logits).all():
                raise RuntimeError('Classifier returned invalid class logits')
            probabilities = logits.softmax(dim=-1)[0].cpu().tolist()
        predicted = int(np.argmax(probabilities))
        return {
            'task': 'classification',
            'model_family': self.model_family,
            'device': str(self.device),
            'checkpoint_file': str(self.checkpoint_file),
            'sample_rate': sample_rate, 'valid_frames': self.valid_frames,
            'duration_seconds': self.valid_frames / sample_rate,
            'model_input_length': self.input_length,
            'padded_frames': self.input_length - self.valid_frames,
            'num_classes': len(self.class_names), 'class_names': self.class_names,
            'predicted_class_index': predicted, 'predicted_label': self.class_names[predicted],
            'confidence': probabilities[predicted], 'confidence_uncalibrated': True,
            'class_probabilities': dict(zip(self.class_names, probabilities)),
            'normalizer_refitted': False,
            'context_used_by_model': self.context_dim > 0,
            'context_schema_version': (
                self.context_normalizer.to_dict()['context_schema_version']
                if self.context_normalizer is not None else None
            ),
            'context_features': list(FEATURE_NAMES) if self.context_dim else [],
            **self.framing_metadata, **self.label_metadata,
        }


# Compatibility for callers using the name proposed for the initial four classes.
FourClassPredictor = WavClassifier


def run_prediction(StateClassifier, ModelConfig, model_family, argv=None):
    parser = argparse.ArgumentParser(
        description='Classify one complete five-second stereo PCM16 WAV using saved weights.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--wav', type=Path, required=True)
    parser.add_argument('--context', type=Path,
                        help='JSON with one raw MachiningContext for multimodal CNN1D-VAE inference')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto',
                        help='auto: use CUDA after a successful probe, otherwise CPU (default)')
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    output = args.output.expanduser().resolve()
    source_paths = [args.checkpoint.expanduser().resolve(), args.wav.expanduser().resolve()]
    if args.context is not None:
        source_paths.append(args.context.expanduser().resolve())
    if output in source_paths:
        parser.error('Output JSON must not overwrite the checkpoint or WAV')
    torch.set_num_threads(args.threads)
    predictor = WavClassifier(args.checkpoint, model_family, StateClassifier, ModelConfig, device=args.device)
    print('Checkpoint loaded successfully.', flush=True)
    context = predictor.load_context(args.context) if args.context is not None else None
    result = predictor.predict_wav(args.wav, context=context)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('-' * 64, flush=True)
    print(f'Device: {predictor.device}', flush=True)
    print(f"Predicted label: {result['predicted_label']}", flush=True)
    print(f"Confidence (uncalibrated softmax): {result['confidence']:.6f}", flush=True)
    for name, value in result['class_probabilities'].items():
        print(f'{name}: {value:.6f}', flush=True)
    if result.get('synthetic_only') is not True:
        print('Class scores are uncalibrated.', flush=True)
    print('-' * 64, flush=True)
    return 0
