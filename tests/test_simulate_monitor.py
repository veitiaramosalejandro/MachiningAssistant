"""Local Monitor request seams with temporary WAVs and a tiny classifier only."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import wave

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from Integration import simulate_monitor as demo
from fourclass_prediction import WavClassifier


def save_wav(path, pcm):
    with wave.open(str(path), 'wb') as handle:
        handle.setframerate(44100); handle.setnchannels(2); handle.setsampwidth(2)
        handle.writeframes(np.asarray(pcm, dtype='<i2').tobytes())


class StubPredictor:
    sample_rate = 44100
    valid_frames = 220500
    class_names = ['quiet', 'changed']
    model_family = 'CNN1D-VAE'
    label_metadata = {'synthetic_only': True}

    def __init__(self, checkpoint):
        self.checkpoint_file = Path(checkpoint)
        self.calls = 0

    def predict_pcm(self, pcm, sample_rate):
        assert pcm.shape == (self.valid_frames, 2) and pcm.dtype == np.float32
        assert sample_rate == self.sample_rate
        self.calls += 1
        return {'predicted_class_index': 0, 'predicted_label': 'quiet', 'confidence': .75,
                'class_probabilities': {'quiet': .75, 'changed': .25},
                'checkpoint_file': str(self.checkpoint_file), 'normalizer_refitted': False,
                'confidence_uncalibrated': True, 'synthetic_only': True}


def test_raw_pcm_decode_preserves_signed_values_and_channel_order():
    values = np.array([[-32768, 32767], [123, -456]], dtype='<i2')
    decoded = demo.decode_request(demo.make_request(2), values.tobytes())
    assert decoded.dtype == np.float32
    np.testing.assert_array_equal(decoded, values)


@pytest.mark.parametrize('field,value', [('sample_rate', 48000), ('channels', 1),
    ('frame_count', 3), ('first_frame', -1), ('sequence_number', True), ('encoding', 'WAV')])
def test_invalid_request_metadata_or_payload_size_is_rejected(field, value):
    request = demo.make_request(2)
    request[field] = value
    with pytest.raises(ValueError):
        demo.decode_request(request, np.zeros((2, 2), dtype='<i2').tobytes())


def test_stream_rejects_sequence_gaps_and_frame_gaps_before_consuming_input(tmp_path):
    engine = StubPredictor(tmp_path/'fake.pt')
    session = demo.LocalAnalysisService(engine).new_stream()
    payload = np.zeros((3, 2), dtype='<i2').tobytes()
    assert session.push(demo.make_request(3), payload) == []
    for request in (demo.make_request(3, 3, 2), demo.make_request(3, 4, 1)):
        with pytest.raises(ValueError, match='Non-contiguous'):
            session.push(request, payload)
        assert (session.next_sequence, session.next_frame, session.stream.input_frames) == (1, 3, 3)
    remainder = np.zeros((demo.WINDOW_FRAMES-3, 2), dtype='<i2').tobytes()
    result, = session.push(demo.make_request(demo.WINDOW_FRAMES-3, 3, 1), remainder)
    assert result['start_frame'] == 0 and result['end_frame'] == demo.WINDOW_FRAMES
    assert engine.calls == 1
    assert session.finish()['trailing_frames'] == 0
    with pytest.raises(RuntimeError, match='finished'):
        session.push(demo.make_request(3, demo.WINDOW_FRAMES, 2), payload)


@dataclass
class TinyConfig:
    sequence_length: int
    input_channels: int = 2


class TinyClassifier(torch.nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.head = torch.nn.Linear(2, num_classes)
        self.calls = 0
        self.first_normalized_values = []

    def forward(self, signal, reconstruct=True):
        assert not reconstruct and not self.training and not torch.is_grad_enabled()
        assert signal.shape == (1, 220544, 2)
        assert torch.count_nonzero(signal[:, demo.WINDOW_FRAMES:]) == 0
        self.calls += 1
        self.first_normalized_values.append(signal[0, 0].tolist())
        return {'logits': self.head(signal[:, :demo.WINDOW_FRAMES].mean(dim=1))}


def test_both_reuses_one_loaded_predictor_and_matches_first_window(tmp_path, capsys):
    config = TinyConfig(220544)
    tiny = TinyClassifier(config, 2)
    checkpoint = tmp_path/'fixture.pt'
    normalizer = {'mean_pcm': [10., -20.], 'scale_pcm': [2., 4.]}
    torch.save({'task': 'classification', 'model_family': 'CNN1D-VAE', 'sample_rate': 44100,
                'valid_frames': demo.WINDOW_FRAMES, 'model_config': asdict(config),
                'class_names': ['quiet', 'changed'], 'num_classes': 2,
                'normalizer': normalizer, 'model_state_dict': tiny.state_dict(),
                'synthetic_only': True}, checkpoint)
    pcm = np.empty((2*demo.WINDOW_FRAMES+17, 2), dtype='<i2')
    pcm[:demo.WINDOW_FRAMES] = [12, -12]
    pcm[demo.WINDOW_FRAMES:2*demo.WINDOW_FRAMES] = [-2, 8]
    pcm[-17:] = [30000, -30000]
    wav = tmp_path/'sample.wav'; save_wav(wav, pcm)
    loaded = []

    def factory(model, path):
        assert model == 'cnn' and path == checkpoint.resolve()
        predictor = WavClassifier(path, 'CNN1D-VAE', TinyClassifier, TinyConfig)
        loaded.append(predictor)
        return predictor

    result = demo.run_simulation('cnn', checkpoint, wav, tmp_path/'result',
                                 mode='both', realtime=False, threads=1, engine_factory=factory)
    assert len(loaded) == result['model_load_count'] == 1
    assert loaded[0].model.calls == result['inference_calls'] == 4
    assert loaded[0].normalizer == normalizer
    assert loaded[0].model.first_normalized_values == [[1., 2.], [-6., 7.], [1., 2.], [-6., 7.]]
    assert result['segment'] is None
    assert result['file']['emitted_windows'] == 2
    assert result['file']['trailing_frames'] == 17
    assert result['file']['first_response']['request']['byte_count'] == 882000
    assert result['stream']['emitted_windows'] == 2
    assert result['stream']['trailing_frames'] == 17
    assert result['stream']['chunk_frames'] == 882
    assert result['first_window_comparison']['matches'] is True
    assert result['first_window_comparison']['max_probability_difference'] == 0
    assert result['normalizer_refitted'] is False and result['confidence_uncalibrated'] is True
    assert result['synthetic_only'] is True and result['network_connection'] is False
    rows = [json.loads(line) for line in (tmp_path/'result'/'responses.jsonl').read_text().splitlines()]
    assert [row['response_kind'] for row in rows] == ['file', 'file', 'stream', 'stream']
    assert [row['start_frame'] for row in rows] == [0, demo.WINDOW_FRAMES, 0, demo.WINDOW_FRAMES]
    assert [row['end_frame'] for row in rows] == [demo.WINDOW_FRAMES, 2*demo.WINDOW_FRAMES]*2
    for offline, streamed in zip(rows[:2], rows[2:]):
        assert offline['class_probabilities'] == streamed['class_probabilities']
    assert all('payload' not in row for row in rows)
    assert json.loads((tmp_path/'result'/'report.json').read_text()) == result
    console = capsys.readouterr().out
    assert 'Transport: local Python calls (no gRPC/C# connection)' in console
    assert 'Mode: infer; Input: both' in console


@pytest.mark.parametrize('mode', ['segment', 'both'])
def test_short_wav_rejects_segment_modes_before_loading_or_writing(tmp_path, mode):
    checkpoint = tmp_path/'fixture.pt'; checkpoint.write_bytes(b'not loaded')
    wav = tmp_path/'short.wav'; save_wav(wav, np.zeros((17, 2)))
    with pytest.raises(ValueError, match='five-second'):
        demo.run_simulation('cnn', checkpoint, wav, tmp_path/'out', mode=mode,
                             engine_factory=lambda *_: pytest.fail('Short WAV must fail before loading'))
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('mode,frames,expected_calls', [
    ('stream', 17, 0), ('segment', 220517, 1), ('file', 17, 0),
    ('file', 2*demo.WINDOW_FRAMES+17, 2),
])
def test_individual_modes_keep_tail_and_segment_scope_separate(tmp_path, mode, frames, expected_calls):
    checkpoint = tmp_path/'fixture.pt'; checkpoint.write_bytes(b'not loaded')
    wav = tmp_path/'sample.wav'; save_wav(wav, np.zeros((frames, 2)))
    engine = StubPredictor(checkpoint)
    report = demo.run_simulation('cnn', checkpoint, wav, tmp_path/'out', mode=mode,
                                 realtime=False, display=False, engine_factory=lambda *_: engine)
    assert engine.calls == expected_calls
    if mode == 'stream':
        assert report['segment'] is None
        assert report['stream']['trailing_frames'] == 17
        assert report['stream']['emitted_windows'] == 0
    elif mode == 'file':
        assert report['segment'] is None and report['stream'] is None
        summary = report['file']
        assert summary['emitted_windows'] == expected_calls
        assert summary['input_frames'] == frames
        assert summary['processed_frames'] == expected_calls*demo.WINDOW_FRAMES
        assert summary['trailing_frames'] == 17
        rows = [json.loads(line) for line in (tmp_path/'out'/'responses.jsonl').read_text().splitlines()]
        assert [row['response_kind'] for row in rows] == ['file']*expected_calls
        assert [(row['start_s'], row['end_s']) for row in rows] == [(i*5., (i+1)*5.) for i in range(expected_calls)]
        if not expected_calls:
            assert summary['first_response'] is None
    else:
        assert report['stream'] is None
        assert report['segment']['remaining_source_frames_not_sent'] == 17


def test_nonempty_output_is_preserved_and_model_is_not_loaded(tmp_path):
    checkpoint = tmp_path/'fixture.pt'; checkpoint.write_bytes(b'not loaded')
    wav = tmp_path/'sample.wav'; save_wav(wav, np.zeros((17, 2)))
    output = tmp_path/'out'; output.mkdir()
    existing = output/'report.json'; existing.write_text('existing user report')
    with pytest.raises(FileExistsError):
        demo.run_simulation('cnn', checkpoint, wav, output, mode='stream',
                             engine_factory=lambda *_: pytest.fail('Must reject before loading'))
    assert existing.read_text() == 'existing user report'


def test_cli_defaults_and_no_realtime_are_forwarded(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(demo, 'run_simulation', lambda *args, **kwargs: calls.append((args, kwargs)))
    required = ['--model', 'lstm', '--checkpoint', str(tmp_path/'best.pt'),
                '--wav', str(tmp_path/'sample.wav'), '--output-dir', str(tmp_path/'out')]
    assert demo.main(required) == 0
    assert calls[0][1] == {'mode': 'both', 'chunk_ms': 20., 'realtime': True, 'threads': 2}
    assert demo.main([*required, '--mode', 'stream', '--no-realtime']) == 0
    assert calls[1][1]['mode'] == 'stream' and calls[1][1]['realtime'] is False
    assert demo.main([*required, '--mode', 'infer', '--input-mode', 'file']) == 0
    assert calls[2][1]['mode'] == 'file'
