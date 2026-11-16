"""Python integration interface for raw stereo PCM windows and streaming chunks."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from data import ChannelNormalizer, read_wav
from model import ModelConfig, VAE
from device_support import resolve_device


MODEL_FAMILY = 'CNN1D-VAE'
SAMPLE_RATE = 44100


def _pcm(value, dimensions):
    x = np.asarray(value, dtype=np.float32)
    if x.ndim != dimensions or x.shape[-1] != 2:
        raise ValueError(f'Expected raw PCM array with {dimensions} dimensions and 2 channels')
    if not np.isfinite(x).all():
        raise ValueError('PCM input contains non-finite values')
    return x


class VAEEngine:
    """Return reconstructions and uncalibrated errors; no fault labels are inferred.

    Input units are raw signed PCM16 counts (not float audio scaled to [-1,1]).
    A new engine uses fixed division by 32768. Checkpoints store their own
    per-channel normalizer, which is always reused without refitting at inference.
    """

    def __init__(self, config=None, normalizer=None, model=None,
                 model_status='untrained', normalizer_provenance='fixed_pcm16_scale', device='auto'):
        self.config = config or ModelConfig()
        if self.config.input_channels != 2:
            raise ValueError('The current integration interface supports two channels')
        self.model = model if model is not None else VAE(self.config)
        self.device = resolve_device(device)
        self.model = self.model.to(self.device).eval()
        self.normalizer = normalizer if normalizer is not None else ChannelNormalizer.full_scale()
        self.model_status = model_status
        self.normalizer_provenance = normalizer_provenance

    def predict_windows(self, pcm, sample_rate=SAMPLE_RATE, include_reconstruction=True):
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f'Expected {SAMPLE_RATE} Hz; input is not automatically resampled')
        x = _pcm(pcm, 3)
        if x.shape[0] == 0 or x.shape[1] != self.config.sequence_length:
            raise ValueError(f'Expected nonempty (batch, {self.config.sequence_length}, 2) input')
        normalized = self.normalizer.transform(x)
        tensor = torch.from_numpy(np.ascontiguousarray(normalized)).to(self.device)
        with torch.inference_mode():
            output = self.model(tensor, sample=False)
            reconstruction = output['reconstruction'].cpu().numpy()
            mu = output['mu'].cpu().numpy()
        if not np.isfinite(reconstruction).all() or not np.isfinite(mu).all():
            raise RuntimeError('Model produced non-finite results')
        per_channel = np.mean((reconstruction-normalized)**2, axis=1)
        result = {
            'model_family': MODEL_FAMILY, 'device': str(self.device),
            'model_status': self.model_status,
            'validated_for_fault_detection': False,
            'sample_rate_hz': SAMPLE_RATE,
            'window_samples': self.config.sequence_length,
            'window_duration_s': self.config.sequence_length/SAMPLE_RATE,
            'score_type': 'reconstruction_mse_standardized',
            'reconstruction_mse_standardized': per_channel.mean(axis=1).tolist(),
            'reconstruction_mse_by_channel_standardized': per_channel.tolist(),
            'latent_mu': mu.tolist(),
        }
        if include_reconstruction:
            result['reconstruction_pcm'] = self.normalizer.inverse(reconstruction).tolist()
        return result

    def predict_window(self, pcm, sample_rate=SAMPLE_RATE, include_reconstruction=True):
        """Single window; result retains the batch dimension of size one."""
        x = _pcm(pcm, 2)
        return self.predict_windows(x[None], sample_rate, include_reconstruction)

    def save(self, path, training_info=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'format_version': 1, 'model_family': MODEL_FAMILY,
            'model_config': asdict(self.config),
            'state_dict': {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
            'normalizer': self.normalizer.to_dict(),
            'normalizer_provenance': self.normalizer_provenance,
            'model_status': self.model_status, 'sample_rate_hz': SAMPLE_RATE,
            'training_info': training_info or {},
        }, path)

    @classmethod
    def from_checkpoint(cls, path, device='auto'):
        state = torch.load(Path(path), map_location='cpu', weights_only=True)
        if state['model_family'] != MODEL_FAMILY:
            raise ValueError('Checkpoint belongs to a different model architecture')
        if state['sample_rate_hz'] != SAMPLE_RATE:
            raise ValueError('Checkpoint sample rate does not match this interface')
        config = ModelConfig(**state['model_config'])
        model = VAE(config)
        model.load_state_dict(state['state_dict'])
        return cls(config, ChannelNormalizer.from_dict(state['normalizer']), model,
                   state['model_status'], state['normalizer_provenance'], device=device)


class StreamingPredictor:
    """Buffer arbitrary chunk lengths and emit complete, nonoverlapping windows.

    Each emitted window is modeled independently, including for LSTM-VAE.
    Residual samples remain buffered; reset() starts a new recording.
    """
    def __init__(self, engine, batch_size=8):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        self.engine = engine
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self.buffer = np.empty((0, 2), dtype=np.float32)
        self.processed_frames = 0

    @property
    def pending_frames(self):
        return len(self.buffer)

    def push(self, pcm_chunk, sample_rate=SAMPLE_RATE):
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f'Expected {SAMPLE_RATE} Hz')
        chunk = _pcm(pcm_chunk, 2)
        self.buffer = np.concatenate((self.buffer, chunk), axis=0)
        length = self.engine.config.sequence_length
        results = []
        while len(self.buffer) >= length:
            count = min(self.batch_size, len(self.buffer)//length)
            batch = self.buffer[:count*length].reshape(count, length, 2)
            prediction = self.engine.predict_windows(batch, include_reconstruction=False)
            for i in range(count):
                start = self.processed_frames + i*length
                results.append({
                    'start_frame': start, 'end_frame': start+length,
                    'start_s': start/SAMPLE_RATE, 'end_s': (start+length)/SAMPLE_RATE,
                    'reconstruction_mse_standardized': prediction['reconstruction_mse_standardized'][i],
                    'reconstruction_mse_by_channel_standardized': prediction['reconstruction_mse_by_channel_standardized'][i],
                    'latent_mu': prediction['latent_mu'][i],
                    'model_status': prediction['model_status'], 'device': str(self.engine.device),
                    'validated_for_fault_detection': False,
                })
            self.processed_frames += count*length
            self.buffer = self.buffer[count*length:].copy()
        return results


def predict_wav(engine, path, max_windows=32, batch_size=8):
    """Read a WAV and process a bounded prefix, or all full windows if limit is 0."""
    if max_windows < 0:
        raise ValueError('max_windows must be nonnegative')
    fs, pcm = read_wav(path)
    length = engine.config.sequence_length
    available = len(pcm)//length
    count = available if max_windows == 0 else min(available, max_windows)
    streamer = StreamingPredictor(engine, batch_size=batch_size)
    rows = []
    for start in range(0, count*length, batch_size*length):
        rows.extend(streamer.push(pcm[start:min(start+batch_size*length, count*length)], fs))
    return {
        'model_family': MODEL_FAMILY, 'model_status': engine.model_status, 'device': str(engine.device),
        'validated_for_fault_detection': False, 'wav_file': str(Path(path)),
        'sample_rate_hz': fs, 'available_full_windows': available,
        'processed_windows': count, 'unprocessed_full_windows': available-count,
        'trailing_partial_frames': len(pcm)%length,
        'score_type': 'reconstruction_mse_standardized', 'windows': rows,
    }
