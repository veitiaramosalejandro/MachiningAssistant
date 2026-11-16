"""Chunked/offline equivalence using small model fixtures, never live weights."""
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
import classification_streaming as streaming
from fourclass_prediction import WavClassifier
from Dataset.Preprocessing import fourclass_data


FRAMES = 220500
NAMES = ['ready', 'cutting', 'change']


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@dataclass
class TinyConfig:
    sequence_length: int
    input_channels: int = 2
    latent_dim: int = 16


class TinyClassifier(torch.nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.head = torch.nn.Linear(2, num_classes)
        self.calls = 0
        self.received_lengths = []

    def forward(self, signal, reconstruct=True):
        assert reconstruct is False and not self.training and not torch.is_grad_enabled()
        self.calls += 1
        self.received_lengths.append(signal.shape[1])
        if signal.shape[1] > FRAMES:
            assert torch.count_nonzero(signal[:, FRAMES:]) == 0
        return {'logits': self.head(signal[:, :FRAMES].mean(dim=1))}


def checkpoint(path, family):
    cfg = TinyConfig(fourclass_data.model_input_length(family))
    model = TinyClassifier(cfg, len(NAMES))
    with torch.no_grad():
        model.head.weight.copy_(torch.tensor([[.1, 0.], [.2, -.05], [.3, -.1]]))
        model.head.bias.copy_(torch.tensor([0., .1, .2]))
    torch.save({'task': 'classification', 'model_family': family, 'model_config': asdict(cfg),
                'sample_rate': 44100, 'valid_frames': FRAMES, 'class_names': NAMES, 'num_classes': len(NAMES),
                'normalizer': {'mean_pcm': [10., -20.], 'scale_pcm': [2., 4.]},
                'model_state_dict': model.state_dict(), 'training_state': {},
                'synthetic_only': True, 'label_semantics': 'Engineering test fixture only'}, path)
    return path


def signal(tail=31):
    values = np.empty((2*FRAMES+tail, 2), dtype=np.int16)
    values[:FRAMES] = [12, -12]
    values[FRAMES:2*FRAMES] = [-2, 8]
    values[2*FRAMES:] = [30000, -30000]
    return values


def save_wav(path, pcm, rate=44100):
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(2); wav.setsampwidth(2); wav.setframerate(rate)
        wav.writeframes(np.asarray(pcm, dtype='<i2').tobytes())


@pytest.mark.parametrize('family,length', [('CNN1D-VAE', 220544), ('LSTM-VAE', FRAMES)])
def test_arbitrary_chunks_match_offline_windows_and_reuse_one_model(tmp_path, family, length):
    path = checkpoint(tmp_path/'fixture.pt', family)
    engine = WavClassifier(path, family, TinyClassifier, TinyConfig)
    pcm = signal()
    expected = [engine.predict_pcm(pcm[start:start+FRAMES]) for start in [0, FRAMES]]
    engine.model.calls = 0
    loaded_model = engine.model
    session = streaming.StreamingClassifier(engine)
    cursor, rows = 0, []
    for size in [0, 17, 127, 220000, 400, 1, 900000]:
        part = pcm[cursor:cursor+size]
        cursor += len(part)
        rows.extend(session.push(part, sample_rate=44100))
    assert cursor == len(pcm)
    assert session.predictor.model is loaded_model and loaded_model.calls == 2
    assert all(actual == length for actual in loaded_model.received_lengths)
    assert len(rows) == 2
    for index, (row, direct) in enumerate(zip(rows, expected)):
        assert row['start_frame'] == index*FRAMES and row['end_frame'] == (index+1)*FRAMES
        assert row['start_s'] == index*5 and row['end_s'] == (index+1)*5
        assert row['sequence_number'] == index
        assert row['class_names'] == NAMES and row['num_classes'] == 3
        assert row['predicted_label'] == direct['predicted_label']
        assert row['class_probabilities'] == direct['class_probabilities']
        assert row['normalizer_refitted'] is False
    summary = session.finish()
    assert summary['emitted_windows'] == 2
    assert summary['processed_frames'] == 2*FRAMES
    assert summary['processed_signal_duration_s'] == 10
    assert summary['trailing_frames'] == 31
    assert summary['input_frames'] == 2*FRAMES+31
    assert session.finish() == summary


def test_finish_never_pads_or_carries_a_tail_into_a_new_session(tmp_path):
    engine = WavClassifier(checkpoint(tmp_path/'fixture.pt', 'CNN1D-VAE'), 'CNN1D-VAE', TinyClassifier, TinyConfig)
    session = streaming.StreamingClassifier(engine)
    assert session.push(np.empty((0, 2), np.int16)) == []
    assert session.finish()['trailing_frames'] == 0
    session.reset()
    assert session.push(np.tile([30000, -30000], (37, 1))) == []
    summary = session.finish()
    assert summary['emitted_windows'] == 0 and summary['trailing_frames'] == 37
    assert engine.model.calls == 0
    new_pcm = np.tile([-2, 8], (FRAMES, 1)).astype(np.int16)
    with pytest.raises(RuntimeError, match='reset'):
        session.push(new_pcm)
    session.reset()
    row, = session.push(new_pcm)
    assert row['start_frame'] == 0 and row['sequence_number'] == 0
    assert row['class_probabilities'] == engine.predict_pcm(new_pcm)['class_probabilities']
    assert session.finish()['trailing_frames'] == 0


def test_buffer_owns_input_and_rejects_bad_chunks_before_mutating_counters(tmp_path):
    engine = WavClassifier(checkpoint(tmp_path/'fixture.pt', 'LSTM-VAE'), 'LSTM-VAE', TinyClassifier, TinyConfig)
    session = streaming.StreamingClassifier(engine)
    original = np.tile([12, -12], (FRAMES, 1)).astype(np.int16)
    chunk = original[:100].copy()
    assert session.push(chunk) == []
    chunk[:] = 30000
    for bad, rate in [(np.zeros((3, 1)), 44100), (np.full((3, 2), np.nan), 44100), (np.zeros((3, 2)), 48000)]:
        with pytest.raises(ValueError):
            session.push(bad, sample_rate=rate)
        assert session.input_frames == session.pending_frames == 100
    row, = session.push(original[100:])
    assert row['class_probabilities'] == engine.predict_pcm(original)['class_probabilities']


@pytest.mark.parametrize('frames,expected,tail', [(FRAMES, 1, 0), (FRAMES-1, 0, FRAMES-1)])
def test_replay_reports_exact_window_and_tail_counts(tmp_path, frames, expected, tail):
    engine = WavClassifier(checkpoint(tmp_path/'fixture.pt', 'CNN1D-VAE'), 'CNN1D-VAE', TinyClassifier, TinyConfig)
    wav, output = tmp_path/'signal.wav', tmp_path/'rows.jsonl'
    save_wav(wav, signal()[:frames])
    seen_after_flush = []
    summary = streaming.replay_stream(engine, wav, output, chunk_ms=20, realtime=False,
                                     on_result=lambda row: seen_after_flush.append(len(output.read_text().splitlines())))
    assert summary['emitted_windows'] == expected and summary['trailing_frames'] == tail
    assert summary['processed_signal_duration_s'] == expected*5
    assert summary['input_frames'] == frames
    assert summary['chunk_frames'] == 882 and summary['realtime'] is False
    assert summary['wall_time_seconds'] >= 0
    assert len(output.read_text().splitlines()) == expected
    assert seen_after_flush == list(range(1, expected+1))
    assert engine.model.calls == expected
    if not expected:
        assert summary['mean_confidence'] is None


def test_realtime_pacing_follows_wav_time_without_a_real_wait(tmp_path, monkeypatch):
    engine = WavClassifier(checkpoint(tmp_path/'fixture.pt', 'CNN1D-VAE'), 'CNN1D-VAE', TinyClassifier, TinyConfig)
    wav, output = tmp_path/'signal.wav', tmp_path/'rows.jsonl'
    save_wav(wav, signal()[:FRAMES+37])

    class Clock:
        now = 100.
        waits = []

        def counter(self):
            return self.now

        def sleep(self, duration):
            assert duration > 0
            self.waits.append(duration)
            self.now += duration

    clock = Clock()
    monkeypatch.setattr(streaming.time, 'perf_counter', clock.counter)
    monkeypatch.setattr(streaming.time, 'sleep', clock.sleep)
    report = streaming.replay_stream(engine, wav, output, realtime=True)
    assert report['emitted_windows'] == 1 and report['trailing_frames'] == 37
    assert report['wall_time_seconds'] == pytest.approx((FRAMES+37)/44100)
    assert sum(clock.waits) == pytest.approx((FRAMES+37)/44100)
    row = json.loads(output.read_text())
    assert row['emitted_after_replay_start_s'] == pytest.approx(5.)
    assert row['latency_after_window_end_ms'] == pytest.approx(0.)


def test_cli_loads_once_and_writes_dynamic_predictions_and_report(tmp_path, monkeypatch, capsys):
    path = checkpoint(tmp_path/'fixture.pt', 'CNN1D-VAE')
    wav, output = tmp_path/'long.wav', tmp_path/'results'/'stream.jsonl'
    save_wav(wav, signal())
    original_load = torch.load
    loads = []

    def counted_load(*args, **kwargs):
        loads.append(args[0])
        return original_load(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError('Streaming must not refit normalization or reload dataset class names')

    monkeypatch.setattr(torch, 'load', counted_load)
    monkeypatch.setattr(fourclass_data, 'fit_normalizer', forbidden)
    monkeypatch.setattr(fourclass_data, 'load_class_names', forbidden)
    assert streaming.run_streaming(TinyClassifier, TinyConfig, 'CNN1D-VAE', [
        '--checkpoint', str(path), '--wav', str(wav), '--output', str(output), '--no-realtime', '--threads', '1',
    ]) == 0
    assert loads == [path.resolve()]
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    report = json.loads(output.with_name('stream_report.json').read_text())
    assert len(rows) == report['emitted_windows'] == 2
    assert report['trailing_frames'] == 31
    assert report['class_names'] == NAMES and report['num_classes'] == 3
    assert report['checkpoint_file'] == str(path.resolve()) and report['normalizer_refitted'] is False
    assert report['realtime'] is False
    assert all(set(row['class_probabilities']) == set(NAMES) for row in rows)
    console = capsys.readouterr().out
    assert console.count('Checkpoint loaded successfully.') == 1
    assert 'Window 0:' in console and 'Window 1:' in console
    assert 'uncalibrated' in console


def test_cli_rejects_output_overwriting_a_source(tmp_path):
    path = checkpoint(tmp_path/'fixture.pt', 'CNN1D-VAE')
    wav = tmp_path/'signal.wav'; save_wav(wav, signal()[:FRAMES])
    before = path.read_bytes()
    with pytest.raises(SystemExit):
        streaming.run_streaming(TinyClassifier, TinyConfig, 'CNN1D-VAE', [
            '--checkpoint', str(path), '--wav', str(wav), '--output', str(path),
        ])
    assert path.read_bytes() == before
