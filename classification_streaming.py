"""Classify complete PCM windows from arbitrary chunks or paced WAV replay.

A session covers one source recording. finish() reports an incomplete tail
without padding it; reset() explicitly starts a separate session using the same
loaded model. This module does not capture microphones or implement transport.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import json
import math
from pathlib import Path
import time
import wave

import numpy as np
import torch

from fourclass_prediction import WavClassifier


class StreamingClassifier:
    """Buffer raw stereo PCM counts and reuse one loaded WavClassifier."""

    def __init__(self, predictor):
        self.predictor = predictor
        self.sample_rate = predictor.sample_rate
        self.valid_frames = predictor.valid_frames
        self.reset()

    def reset(self):
        self._chunks = deque()
        self.pending_frames = 0
        self.input_frames = 0
        self.processed_frames = 0
        self.emitted_windows = 0
        self._finished = False

    def _next_window(self):
        window = np.empty((self.valid_frames, 2), dtype=np.float32)
        copied = 0
        for chunk in self._chunks:
            count = min(len(chunk), self.valid_frames-copied)
            window[copied:copied+count] = chunk[:count]
            copied += count
            if copied == self.valid_frames:
                break
        return window

    def _consume_window(self):
        remaining = self.valid_frames
        while remaining:
            chunk = self._chunks[0]
            count = min(len(chunk), remaining)
            if count == len(chunk):
                self._chunks.popleft()
            else:
                self._chunks[0] = chunk[count:]
            remaining -= count
        self.pending_frames -= self.valid_frames

    def push(self, pcm, sample_rate=None, context_provider=None, window_callback=None):
        if self._finished:
            raise RuntimeError('Session is finished; call reset() or create a new session before another recording')
        if sample_rate is not None and sample_rate != self.sample_rate:
            raise ValueError(f'Expected {self.sample_rate} Hz; streaming input is not resampled')
        values = np.asarray(pcm, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError('Expected raw stereo PCM counts with shape (frames, 2)')
        if not np.isfinite(values).all():
            raise ValueError('PCM chunk contains non-finite values')
        if not len(values):
            return []
        # Own buffered data so a caller can reuse its capture buffer safely.
        self._chunks.append(np.ascontiguousarray(values).copy())
        self.input_frames += len(values)
        self.pending_frames += len(values)
        results = []
        while self.pending_frames >= self.valid_frames:
            window = self._next_window()
            start = self.processed_frames
            end = start+self.valid_frames
            window_artifact = None
            if window_callback is not None:
                window_artifact = window_callback(
                    window, self.emitted_windows, start / self.sample_rate, end / self.sample_rate
                )
            context = None if context_provider is None else context_provider(
                start / self.sample_rate, end / self.sample_rate
            )
            prediction = self.predictor.predict_pcm(
                window, sample_rate=self.sample_rate, context=context
            )
            result = {**prediction, 'sequence_number': self.emitted_windows,
                      'start_frame': start, 'end_frame': end,
                      'start_s': start/self.sample_rate, 'end_s': end/self.sample_rate}
            if window_artifact is not None:
                result['audio_window_file'] = str(window_artifact)
            results.append(result)
            # Commit consumption after successful inference; no input is lost
            # merely because a caller catches an inference exception.
            self._consume_window()
            self.processed_frames = end
            self.emitted_windows += 1
        return results

    def finish(self):
        """Close the session and report, but never classify, a short tail."""
        self._finished = True
        return {'sample_rate_hz': self.sample_rate, 'window_frames': self.valid_frames,
                'window_duration_s': self.valid_frames/self.sample_rate,
                'input_frames': self.input_frames, 'input_duration_s': self.input_frames/self.sample_rate,
                'emitted_windows': self.emitted_windows, 'processed_frames': self.processed_frames,
                'processed_signal_duration_s': self.processed_frames/self.sample_rate,
                'trailing_frames': self.pending_frames, 'trailing_duration_s': self.pending_frames/self.sample_rate,
                'tail_policy': 'not padded or inferred; never joined to another recording',
                'session_finished': True}


def replay_stream(predictor, wav_path, output_path, chunk_ms=20., realtime=False, on_result=None):
    """Read one WAV incrementally and flush each completed prediction as JSONL."""
    if not math.isfinite(chunk_ms) or chunk_ms <= 0:
        raise ValueError('chunk_ms must be finite and positive')
    chunk_frames = round(predictor.sample_rate*chunk_ms/1000)
    if chunk_frames < 1:
        raise ValueError('chunk_ms must cover at least one audio frame')
    wav_path = Path(wav_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path in (wav_path, predictor.checkpoint_file):
        raise ValueError('Stream output must not overwrite the input WAV or checkpoint')
    streamer = StreamingClassifier(predictor)
    counts = Counter({name: 0 for name in predictor.class_names})
    confidence_sum = 0.
    received_chunks = 0
    with wave.open(str(wav_path), 'rb') as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth(), source.getcomptype()) != (
            predictor.sample_rate, 2, 2, 'NONE',
        ):
            raise ValueError('Expected 44100 Hz, stereo, uncompressed PCM16 WAV')
        expected_frames = source.getnframes()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with output_path.open('w', encoding='utf-8') as writer:
            while True:
                payload = source.readframes(chunk_frames)
                if not payload:
                    break
                if len(payload) % 4:
                    raise ValueError('WAV contains an incomplete stereo PCM frame')
                pcm = np.frombuffer(payload, dtype='<i2').reshape(-1, 2)
                due_s = (streamer.input_frames+len(pcm))/predictor.sample_rate
                if realtime:
                    delay = started+due_s-time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                results = streamer.push(pcm, sample_rate=predictor.sample_rate)
                received_chunks += 1
                finished = time.perf_counter()
                for result in results:
                    row = {**result, 'wav_file': str(wav_path),
                           'source_available_through_s': due_s,
                           'emitted_after_replay_start_s': finished-started,
                           'latency_after_window_end_ms': (finished-started-result['end_s'])*1000 if realtime else None}
                    writer.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                    writer.flush()
                    counts[row['predicted_label']] += 1
                    confidence_sum += row['confidence']
                    if on_result is not None:
                        on_result(row)
        wall_time = time.perf_counter()-started
    if streamer.input_frames != expected_frames:
        raise ValueError('Read frame count differs from WAV header')
    summary = streamer.finish()
    if summary['input_frames'] != summary['processed_frames']+summary['trailing_frames']:
        raise AssertionError('Streaming frame accounting mismatch')
    return {**summary, 'task': 'classification', 'model_family': predictor.model_family,
            'checkpoint_file': str(predictor.checkpoint_file), 'wav_file': str(wav_path),
            'output_file': str(output_path), 'mode': 'realtime_wav_replay' if realtime else 'fast_chunk_replay',
            'realtime': bool(realtime), 'requested_chunk_ms': chunk_ms, 'chunk_frames': chunk_frames,
            'chunk_duration_ms': chunk_frames/predictor.sample_rate*1000, 'received_chunks': received_chunks,
            'wall_time_seconds': wall_time, 'class_names': predictor.class_names,
            'num_classes': len(predictor.class_names), 'predicted_label_counts': dict(counts),
            'device': str(predictor.device),
            'mean_confidence': confidence_sum/summary['emitted_windows'] if summary['emitted_windows'] else None,
            'confidence_uncalibrated': True, 'normalizer_refitted': False,
            'session_policy': 'one WAV per session; model runs independently for each complete window',
            **predictor.label_metadata}


def run_streaming(StateClassifier, ModelConfig, model_family, argv=None):
    parser = argparse.ArgumentParser(description='Classify complete five-second windows from stereo PCM16 WAV chunks.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--wav', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--chunk-ms', type=float, default=20.)
    parser.add_argument('--realtime', action=argparse.BooleanOptionalAction, default=False,
                        help='Pace chunks by WAV time; standalone scripts default to fast replay')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto',
                        help='auto: use CUDA after a successful probe, otherwise CPU (default)')
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    if not math.isfinite(args.chunk_ms) or args.chunk_ms <= 0:
        parser.error('--chunk-ms must be finite and positive')
    checkpoint = args.checkpoint.expanduser().resolve()
    wav_path = args.wav.expanduser().resolve()
    for name, path in [('Checkpoint', checkpoint), ('WAV', wav_path)]:
        if not path.is_file():
            parser.error(f'{name} file not found: {path}')
    output = args.output.expanduser().resolve()
    report = args.report.expanduser().resolve() if args.report is not None else output.with_name(output.stem+'_report.json')
    if output in (checkpoint, wav_path) or report in (checkpoint, wav_path, output):
        parser.error('Output and report must be distinct and must not overwrite the checkpoint or WAV')
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    predictor = WavClassifier(checkpoint, model_family, StateClassifier, ModelConfig, device=args.device)
    model_load_seconds = time.perf_counter()-started
    print('Checkpoint loaded successfully.', flush=True)
    print(f'Device: {predictor.device}', flush=True)
    print(f'Replay: {"real-time" if args.realtime else "fast"} WAV chunks; window duration: {predictor.valid_frames/predictor.sample_rate:.3f} s.', flush=True)
    print('-'*64, flush=True)

    def display(row):
        print(f"Window {row['sequence_number']}: [{row['start_s']:.3f}, {row['end_s']:.3f}) s; "
              f"label={row['predicted_label']}; confidence={row['confidence']:.6f}", flush=True)

    summary = replay_stream(predictor, wav_path, output, args.chunk_ms, args.realtime, display)
    summary['model_load_seconds'] = model_load_seconds
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('-'*64, flush=True)
    print(f"Emitted windows: {summary['emitted_windows']}; classified signal duration: {summary['processed_signal_duration_s']:.3f} s.", flush=True)
    print(f"Trailing frames: {summary['trailing_frames']}; replay wall time: {summary['wall_time_seconds']:.3f} s.", flush=True)
    print('Class scores are uncalibrated.', flush=True)
    print('-'*64, flush=True)
    return 0
