"""Single-WAV classifier loading/inference with tiny synthetic model fixtures."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import wave

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fourclass_prediction as prediction
from Dataset.Preprocessing import fourclass_data
from Dataset.Preprocessing.cnc_features import CncFeatureNormalizer, FEATURE_NAMES


DEFAULT_NAMES = ['good', 'sufficient', 'bad', 'very_bad']


@dataclass
class TinyConfig:
    sequence_length: int
    input_channels: int = 2
    latent_dim: int = 16


class TinyClassifier(torch.nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.head = torch.nn.Linear(config.input_channels, num_classes)
        self.received = None

    def forward(self, values, reconstruct=True):
        assert reconstruct is False
        assert self.training is False
        assert not torch.is_grad_enabled()
        self.received = values.detach().clone()
        features = values[:, :fourclass_data.VALID_FRAMES].mean(dim=1)
        return {'logits': self.head(features)}


class TinyMultimodalClassifier(torch.nn.Module):
    def __init__(self, config, num_classes, context_dim, context_embedding_dim):
        super().__init__()
        self.head = torch.nn.Linear(config.input_channels + context_dim, num_classes)
        self.received_context = None

    def forward(self, values, reconstruct=True, *, context=None):
        assert reconstruct is False
        assert context is not None and context.shape[0] == values.shape[0]
        self.received_context = context.detach().clone()
        features = values[:, :fourclass_data.VALID_FRAMES].mean(dim=1)
        return {'logits': self.head(torch.cat((features, context), dim=-1))}


def save_checkpoint(path, family='CNN1D-VAE', names=None, *, synthetic=True):
    names = DEFAULT_NAMES if names is None else names
    config = TinyConfig(fourclass_data.model_input_length(family))
    model = TinyClassifier(config, len(names))
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.copy_(torch.arange(len(names), dtype=torch.float32))
    payload = {
        'task': 'classification', 'model_family': family, 'model_config': asdict(config),
        'sample_rate': 44100, 'valid_frames': 220500, 'class_names': names,
        'num_classes': len(names), 'normalizer': {'mean_pcm': [10., -20.], 'scale_pcm': [2., 4.]},
        'model_state_dict': model.state_dict(), 'training_state': {},
    }
    if synthetic:
        payload.update(synthetic_only=True, label_semantics='Artificial sine-amplitude tiers, not real fault truth.')
    torch.save(payload, path)
    return payload


def context_payload():
    return {
        'SchemaVersion': 1, 'ContextId': 'ctx-1', 'MachineId': 'machine-1',
        'DriverSessionId': 'session-1', 'OccurredAtUtc': '2026-01-01T00:00:00Z',
        'ActualSpindleRpm': 8000, 'ActualFeedMmPerMin': 1200,
        'SPowerPercent': 40, 'WPowerPercent': 20,
        'ToolDiameterMm': 10, 'ToolTeethCount': 4,
        'ToolRemainingUsePercent': 90,
    }


def save_multimodal_checkpoint(path):
    names = DEFAULT_NAMES
    config = TinyConfig(fourclass_data.model_input_length('CNN1D-VAE'))
    normalizer = CncFeatureNormalizer.fit_contexts([context_payload(), {
        **context_payload(), 'ContextId': 'ctx-2', 'ActualSpindleRpm': 9000,
    }])
    model = TinyMultimodalClassifier(config, len(names), len(FEATURE_NAMES), 8)
    torch.save({
        'task': 'classification', 'model_family': 'CNN1D-VAE', 'model_config': asdict(config),
        'sample_rate': 44100, 'valid_frames': 220500, 'class_names': names,
        'num_classes': len(names), 'normalizer': {'mean_pcm': [10., -20.], 'scale_pcm': [2., 4.]},
        'context_dim': len(FEATURE_NAMES), 'context_embedding_dim': 8,
        'context_normalizer': normalizer.to_dict(), 'model_state_dict': model.state_dict(),
    }, path)


def save_wav(path, frames=220500, sample_rate=44100, channels=2):
    pcm = np.empty((frames, channels), dtype='<i2')
    pcm[:, 0] = 12
    if channels == 2:
        pcm[:, 1] = -12
    with wave.open(str(path), 'wb') as wav:
        wav.setframerate(sample_rate); wav.setnchannels(channels); wav.setsampwidth(2)
        wav.writeframes(pcm.tobytes())


@pytest.mark.parametrize('family,length,padding', [('CNN1D-VAE', 220544, 44), ('LSTM-VAE', 220500, 0)])
def test_full_wav_reuses_saved_normalizer_and_shared_padding(tmp_path, monkeypatch, family, length, padding):
    checkpoint, wav = tmp_path/'saved.pt', tmp_path/'named_good.wav'
    save_checkpoint(checkpoint, family)
    save_wav(wav)

    def forbidden(*args, **kwargs):
        raise AssertionError('Prediction must not fit statistics or reload current class labels')

    monkeypatch.setattr(fourclass_data, 'fit_normalizer', forbidden)
    monkeypatch.setattr(fourclass_data, 'load_class_names', forbidden, raising=False)
    engine = prediction.WavClassifier(checkpoint, family, TinyClassifier, TinyConfig)
    result = engine.predict_wav(wav)
    assert tuple(engine.model.received.shape) == (1, length, 2)
    np.testing.assert_array_equal(engine.model.received[0, :220500, 0].numpy(), np.ones(220500))
    np.testing.assert_array_equal(engine.model.received[0, :220500, 1].numpy(), np.full(220500, 2.))
    if padding:
        assert torch.count_nonzero(engine.model.received[0, 220500:]) == 0
    assert result['padded_frames'] == padding and result['duration_seconds'] == 5.0
    assert result['class_names'] == DEFAULT_NAMES
    assert result['predicted_label'] == 'very_bad'  # waveform filename is not a feature
    assert result['predicted_class_index'] == 3
    assert result['confidence'] == max(result['class_probabilities'].values())
    assert sum(result['class_probabilities'].values()) == pytest.approx(1., abs=1e-7)
    assert result['confidence_uncalibrated'] is True and result['normalizer_refitted'] is False
    assert result['synthetic_only'] is True


def test_checkpoint_class_count_and_order_are_preserved_for_other_labels(tmp_path):
    checkpoint, wav = tmp_path/'three_classes.pt', tmp_path/'signal.wav'
    names = ['state_c', 'state_a', 'state_b']
    save_checkpoint(checkpoint, names=names, synthetic=False)
    save_wav(wav)
    engine = prediction.WavClassifier(checkpoint, 'CNN1D-VAE', TinyClassifier, TinyConfig)
    result = engine.predict_wav(wav)
    assert engine.model.head.out_features == 3
    assert result['class_names'] == names
    assert list(result['class_probabilities']) == names
    assert result['num_classes'] == 3 and result['predicted_label'] == 'state_b'
    assert 'synthetic_only' not in result and 'label_semantics' not in result


def test_pcm_and_wav_share_preprocessing_and_predictions(tmp_path):
    checkpoint, wav = tmp_path/'saved.pt', tmp_path/'sample.wav'
    save_checkpoint(checkpoint)
    save_wav(wav)
    engine = prediction.WavClassifier(checkpoint, 'CNN1D-VAE', TinyClassifier, TinyConfig)
    sample_rate, pcm = fourclass_data.read_wav(wav)
    from_file = engine.predict_wav(wav)
    from_pcm = engine.predict_pcm(pcm, sample_rate)
    assert from_file == {**from_pcm, 'wav_file': str(wav.resolve())}
    assert engine.sample_rate == 44100 and engine.valid_frames == 220500
    with pytest.raises(ValueError, match='sample rate'):
        engine.predict_pcm(pcm, sample_rate=48000)


def test_multimodal_prediction_normalizes_context_and_reports_schema(tmp_path):
    checkpoint, wav = tmp_path/'multimodal.pt', tmp_path/'sample.wav'
    save_multimodal_checkpoint(checkpoint)
    save_wav(wav)
    engine = prediction.WavClassifier(
        checkpoint, 'CNN1D-VAE', TinyMultimodalClassifier, TinyConfig, device='cpu'
    )
    result = engine.predict_wav(wav, context=context_payload())
    assert result['context_used_by_model'] is True
    assert result['context_schema_version'] == 2
    assert result['context_features'] == list(FEATURE_NAMES)
    assert engine.model.received_context.shape == (1, len(FEATURE_NAMES))
    assert torch.isfinite(engine.model.received_context).all()
    with pytest.raises(ValueError, match='required'):
        engine.predict_pcm(np.zeros((220500, 2), dtype=np.float32))


@pytest.mark.parametrize('change,match', [
    ({'task': 'reconstruction'}, 'classification checkpoint'),
    ({'model_family': 'LSTM-VAE'}, 'model_family'),
    ({'valid_frames': 2048}, '220500'),
    ({'sample_rate': 48000}, '44100'),
    ({'class_names': ['same', 'same']}, 'class_names'),
    ({'class_names': ['good', '']}, 'class_names'),
    ({'num_classes': 3}, 'num_classes'),
    ({'model_config': {'sequence_length': 2048, 'input_channels': 2, 'latent_dim': 16}}, 'input shape'),
    ({'normalizer': {'mean_pcm': [0., 0.], 'scale_pcm': [0., 1.]}}, 'positive scales'),
])
def test_incompatible_checkpoints_are_rejected(tmp_path, change, match):
    path = tmp_path/'checkpoint.pt'
    payload = save_checkpoint(path)
    payload.update(change)
    torch.save(payload, path)
    with pytest.raises(ValueError, match=match):
        prediction.WavClassifier(path, 'CNN1D-VAE', TinyClassifier, TinyConfig)


def test_legacy_fourclass_task_keeps_saved_mapping(tmp_path):
    path = tmp_path/'legacy_fourclass.pt'
    payload = save_checkpoint(path)
    payload['task'] = 'fourclass'
    del payload['num_classes']
    torch.save(payload, path)
    engine = prediction.WavClassifier(path, 'CNN1D-VAE', TinyClassifier, TinyConfig)
    assert engine.class_names == DEFAULT_NAMES and engine.model.head.out_features == 4


@pytest.mark.parametrize('frames,rate,channels', [(220499, 44100, 2), (441000, 44100, 2), (220500, 48000, 2), (220500, 44100, 1)])
def test_wrong_length_or_wav_format_is_not_cropped_or_resampled(tmp_path, frames, rate, channels):
    checkpoint, wav = tmp_path/'saved.pt', tmp_path/'invalid.wav'
    save_checkpoint(checkpoint)
    save_wav(wav, frames, rate, channels)
    engine = prediction.WavClassifier(checkpoint, 'CNN1D-VAE', TinyClassifier, TinyConfig)
    with pytest.raises(ValueError):
        engine.predict_wav(wav)
    assert engine.model.received is None


def test_cli_writes_dynamic_probabilities_and_concise_english_output(tmp_path, capsys):
    checkpoint, wav, output = tmp_path/'saved weights.pt', tmp_path/'input sound.wav', tmp_path/'results'/'prediction.json'
    save_checkpoint(checkpoint)
    save_wav(wav)
    assert prediction.run_prediction(TinyClassifier, TinyConfig, 'CNN1D-VAE', [
        '--checkpoint', str(checkpoint), '--wav', str(wav), '--output', str(output), '--threads', '1',
    ]) == 0
    result = json.loads(output.read_text())
    assert result['checkpoint_file'] == str(checkpoint.resolve())
    assert result['predicted_label'] == 'very_bad'
    text = capsys.readouterr().out
    assert text.startswith('Checkpoint loaded successfully.\n' + '-'*64)
    assert text.rstrip().endswith('-'*64)
    for name in DEFAULT_NAMES:
        assert f'{name}:' in text
    assert 'uncalibrated softmax' in text
    assert 'Synthetic signal classes; scores are not real fault or machining-quality probabilities.' not in text


def test_cli_does_not_overwrite_source_checkpoint(tmp_path):
    checkpoint, wav = tmp_path/'saved.pt', tmp_path/'input.wav'
    save_checkpoint(checkpoint); save_wav(wav)
    original = checkpoint.read_bytes()
    with pytest.raises(SystemExit) as raised:
        prediction.run_prediction(TinyClassifier, TinyConfig, 'CNN1D-VAE', [
            '--checkpoint', str(checkpoint), '--wav', str(wav), '--output', str(checkpoint),
        ])
    assert raised.value.code == 2 and checkpoint.read_bytes() == original
