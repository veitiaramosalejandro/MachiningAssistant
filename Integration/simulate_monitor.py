"""Demonstrate Monitor-to-AI requests with local calls and real saved inference.

This is a WAV replay adapter, not a gRPC server or a C# integration. Requests
carry decoded interleaved PCM16LE bytes, never a WAV/RIFF container header.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import wave

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classification_streaming import StreamingClassifier
from fourclass_prediction import WavClassifier


TRANSPORT = 'local Python calls (no gRPC/C# connection)'
FAMILIES = {'cnn': 'CNN1D-VAE', 'lstm': 'LSTM-VAE'}
SAMPLE_RATE = 44100
CHANNELS = 2
WINDOW_FRAMES = 220500
FRAME_BYTES = 4
ENCODING = 's16le_interleaved'


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_engine(model, checkpoint, device='auto'):
    """Load the selected family once; restore the shared 'model' import binding."""
    family = FAMILIES[model]
    folder = PROJECT_ROOT/family
    model_module = _load_module(f'_monitor_{model}_model', folder/'model.py')
    previous = sys.modules.get('model')
    try:
        sys.modules['model'] = model_module
        classifier = _load_module(f'_monitor_{model}_classifier', folder/'classifier.py')
    finally:
        if previous is None:
            sys.modules.pop('model', None)
        else:
            sys.modules['model'] = previous
    return WavClassifier(checkpoint, family, classifier.StateClassifier, model_module.ModelConfig, device=device)


def make_request(frame_count, first_frame=0, sequence_number=0):
    return {'encoding': ENCODING, 'sample_rate': SAMPLE_RATE, 'channels': CHANNELS,
            'frame_count': frame_count, 'first_frame': first_frame,
            'sequence_number': sequence_number}


def decode_request(request, payload):
    """Validate metadata and decode raw PCM counts without rescaling or fitting."""
    if request.get('encoding') != ENCODING:
        raise ValueError(f'Expected encoding {ENCODING}')
    if request.get('sample_rate') != SAMPLE_RATE or request.get('channels') != CHANNELS:
        raise ValueError('Expected sample_rate=44100 and channels=2')
    for field in ('frame_count', 'first_frame', 'sequence_number'):
        value = request.get(field)
        if type(value) is not int or value < (1 if field == 'frame_count' else 0):
            raise ValueError(f'{field} must be an integer in the supported nonnegative range')
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError('PCM payload must contain bytes')
    if len(payload) != request['frame_count']*FRAME_BYTES:
        raise ValueError('PCM byte count differs from frame_count * channels * 2')
    return np.frombuffer(payload, dtype='<i2').reshape(-1, CHANNELS).astype(np.float32)


def request_summary(request, payload):
    return {key: request[key] for key in ('encoding', 'sample_rate', 'channels',
            'frame_count', 'first_frame', 'sequence_number')} | {'byte_count': len(payload)}


class LocalAnalysisService:
    """A local AI-service boundary sharing one already-loaded classifier."""

    def __init__(self, predictor):
        if predictor.sample_rate != SAMPLE_RATE or predictor.valid_frames != WINDOW_FRAMES:
            raise ValueError('This demonstration requires a 44100 Hz, five-second classifier')
        self.predictor = predictor

    def analyze_segment(self, request, payload):
        pcm = decode_request(request, payload)
        if request['frame_count'] != self.predictor.valid_frames:
            raise ValueError('Segment requests require one complete five-second signal')
        started = time.perf_counter()
        prediction = self.predictor.predict_pcm(pcm, sample_rate=request['sample_rate'])
        start = request['first_frame']
        end = start+len(pcm)
        return {**prediction, 'response_kind': 'segment',
                'request': request_summary(request, payload),
                'start_frame': start, 'end_frame': end,
                'start_s': start/SAMPLE_RATE, 'end_s': end/SAMPLE_RATE,
                'inference_seconds': time.perf_counter()-started}

    def new_stream(self):
        return LocalStreamSession(self.predictor)


class LocalStreamSession:
    """Validate one recording's ordered requests before using the existing buffer."""

    def __init__(self, predictor):
        self.stream = StreamingClassifier(predictor)
        self.next_sequence = 0
        self.next_frame = 0
        self.finished = False

    def push(self, request, payload):
        if self.finished:
            raise RuntimeError('Session is finished; start a new session for another recording')
        pcm = decode_request(request, payload)
        if request['sequence_number'] != self.next_sequence:
            raise ValueError(f'Non-contiguous sequence_number: expected {self.next_sequence}')
        if request['first_frame'] != self.next_frame:
            raise ValueError(f'Non-contiguous first_frame: expected {self.next_frame}')
        started = time.perf_counter()
        results = self.stream.push(pcm, sample_rate=request['sample_rate'])
        elapsed = time.perf_counter()-started
        self.next_frame += len(pcm)
        self.next_sequence += 1
        return [{**result, 'response_kind': 'stream',
                 'request': request_summary(request, payload),
                 'source_available_through_frame': self.next_frame,
                 'source_available_through_s': self.next_frame/SAMPLE_RATE,
                 'service_call_seconds': elapsed} for result in results]

    def finish(self):
        self.finished = True
        return {**self.stream.finish(), 'received_chunks': self.next_sequence}


def _wav_info(path):
    with wave.open(str(path), 'rb') as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth(), source.getcomptype()) != (
                SAMPLE_RATE, CHANNELS, 2, 'NONE'):
            raise ValueError('Expected 44100 Hz, stereo, uncompressed PCM16 WAV')
        frames = source.getnframes()
    return {'sample_rate': SAMPLE_RATE, 'channels': CHANNELS, 'bits_per_sample': 16,
            'frame_count': frames, 'duration_seconds': frames/SAMPLE_RATE}


def _print_request(kind, summary):
    print(f"{kind} request: first_frame={summary['first_frame']}; frames={summary['frame_count']}; "
          f"rate={summary['sample_rate']} Hz; channels={summary['channels']}; bytes={summary['byte_count']}", flush=True)


def _print_result(result):
    print(f"{result['response_kind'].title()} result [{result['start_s']:.3f}, {result['end_s']:.3f}) s: "
          f"label={result['predicted_label']}; uncalibrated score={result['confidence']:.6f}", flush=True)
    print('Probabilities: '+', '.join(f'{name}={value:.6f}' for name, value in result['class_probabilities'].items()), flush=True)


class MonitorSimulator:
    """Read WAV container audio and send only PCM bytes to the local service."""

    def __init__(self, service, wav_path):
        self.service = service
        self.wav_path = Path(wav_path)

    def segment(self):
        with wave.open(str(self.wav_path), 'rb') as source:
            payload = source.readframes(WINDOW_FRAMES)
        request = make_request(WINDOW_FRAMES)
        return self.service.analyze_segment(request, payload)

    def file(self, on_result):
        """Analyze every complete window of a WAV without real-time pacing."""
        input_frames = processed_frames = emitted_windows = 0
        first_result = None
        counts = Counter({name: 0 for name in self.service.predictor.class_names})
        started = time.perf_counter()
        with wave.open(str(self.wav_path), 'rb') as source:
            while True:
                payload = source.readframes(WINDOW_FRAMES)
                if not payload:
                    break
                if len(payload) % FRAME_BYTES:
                    raise ValueError('WAV contains an incomplete stereo PCM frame')
                frames = len(payload)//FRAME_BYTES
                input_frames += frames
                if frames < WINDOW_FRAMES:
                    break
                request = make_request(frames, processed_frames, emitted_windows)
                result = {**self.service.analyze_segment(request, payload),
                          'response_kind': 'file', 'sequence_number': emitted_windows}
                if first_result is None:
                    first_result = result
                counts[result['predicted_label']] += 1
                on_result(result)
                processed_frames += frames
                emitted_windows += 1
        if input_frames != _wav_info(self.wav_path)['frame_count']:
            raise ValueError('Read frame count differs from WAV header')
        trailing_frames = input_frames-processed_frames
        return {'scope': 'entire_WAV_from_frame_zero', 'first_frame': 0,
                'sample_rate_hz': SAMPLE_RATE, 'window_frames': WINDOW_FRAMES,
                'window_duration_s': WINDOW_FRAMES/SAMPLE_RATE,
                'input_frames': input_frames, 'input_duration_s': input_frames/SAMPLE_RATE,
                'emitted_windows': emitted_windows, 'processed_frames': processed_frames,
                'processed_signal_duration_s': processed_frames/SAMPLE_RATE,
                'trailing_frames': trailing_frames, 'trailing_duration_s': trailing_frames/SAMPLE_RATE,
                'tail_policy': 'not padded or inferred; never joined to another recording',
                'realtime': False, 'wall_time_seconds': time.perf_counter()-started,
                'predicted_label_counts': dict(counts), 'first_response': first_result}

    def stream(self, chunk_frames, realtime, on_result, display=False):
        session = self.service.new_stream()
        first_result = None
        counts = Counter({name: 0 for name in self.service.predictor.class_names})
        started = time.perf_counter()
        with wave.open(str(self.wav_path), 'rb') as source:
            while True:
                payload = source.readframes(chunk_frames)
                if not payload:
                    break
                if len(payload) % FRAME_BYTES:
                    raise ValueError('WAV contains an incomplete stereo PCM frame')
                request = make_request(len(payload)//FRAME_BYTES, session.next_frame, session.next_sequence)
                if display and session.next_sequence == 0:
                    _print_request('First stream chunk', request_summary(request, payload))
                due = (session.next_frame+request['frame_count'])/SAMPLE_RATE
                if realtime:
                    delay = started+due-time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                for result in session.push(request, payload):
                    result['emitted_after_replay_start_s'] = time.perf_counter()-started
                    result['latency_after_window_end_ms'] = (
                        (result['emitted_after_replay_start_s']-result['end_s'])*1000 if realtime else None)
                    if first_result is None:
                        first_result = result
                    counts[result['predicted_label']] += 1
                    on_result(result)
        summary = session.finish()
        if summary['input_frames'] != _wav_info(self.wav_path)['frame_count']:
            raise ValueError('Read frame count differs from WAV header')
        return {**summary, 'scope': 'entire_WAV_from_frame_zero', 'first_frame': 0,
                'realtime': bool(realtime), 'chunk_frames': chunk_frames,
                'chunk_duration_ms': chunk_frames/SAMPLE_RATE*1000,
                'wall_time_seconds': time.perf_counter()-started,
                'predicted_label_counts': dict(counts), 'first_response': first_result}


def run_simulation(model, checkpoint, wav_path, output_dir, mode='both', chunk_ms=20.,
                   realtime=True, threads=2, *, engine_factory=None, display=True, device='auto'):
    if model not in FAMILIES or mode not in ('file', 'segment', 'stream', 'both'):
        raise ValueError('Unsupported model or simulation mode')
    if type(threads) is not int or threads < 1:
        raise ValueError('threads must be a positive integer')
    if not math.isfinite(chunk_ms) or chunk_ms <= 0 or round(SAMPLE_RATE*chunk_ms/1000) < 1:
        raise ValueError('chunk_ms must be finite, positive and cover at least one audio frame')
    checkpoint = Path(checkpoint).expanduser().resolve()
    wav_path = Path(wav_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    for path in (checkpoint, wav_path):
        if not path.is_file():
            raise FileNotFoundError(f'Input file not found: {path}')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Output directory must be new or empty: {output}')
    info = _wav_info(wav_path)
    if mode in ('segment', 'both') and info['frame_count'] < WINDOW_FRAMES:
        raise ValueError('Segment/both mode requires at least one complete five-second window')
    torch.set_num_threads(threads)
    load_started = time.perf_counter()
    engine = engine_factory(model, checkpoint) if engine_factory is not None else load_engine(model, checkpoint, device=device)
    load_seconds = time.perf_counter()-load_started
    service = LocalAnalysisService(engine)
    monitor = MonitorSimulator(service, wav_path)
    output.mkdir(parents=True, exist_ok=True)
    responses_path = output/'responses.jsonl'
    report_path = output/'report.json'
    metadata = dict(getattr(engine, 'label_metadata', {}))
    report = {'schema_version': 1, 'transport': TRANSPORT, 'network_connection': False,
              'input_source': 'pre-recorded WAV decoded to PCM16LE; no RIFF header is sent',
              'model': model, 'model_family': engine.model_family,
              'device': str(getattr(engine, 'device', 'cpu')),
              'checkpoint_file': str(checkpoint), 'source_wav': str(wav_path),
              'source': info, 'mode': mode, 'model_load_count': 1, 'model_load_seconds': load_seconds,
              'normalizer_source': 'saved checkpoint', 'normalizer_refitted': False,
              'confidence_uncalibrated': True, 'synthetic_only': metadata.get('synthetic_only'),
              'class_names': engine.class_names, 'threads': threads,
              'requested_chunk_ms': chunk_ms, 'responses_file': str(responses_path),
              'inference_only': True, 'segment': None, 'file': None, 'stream': None,
              'first_window_comparison': None, **metadata}
    if display:
        print('-'*64, flush=True)
        print('Machining Monitor / AI service simulation', flush=True)
        print(f'Transport: {TRANSPORT}', flush=True)
        print(f"Checkpoint loaded successfully (once, {report['device']}).", flush=True)
        print(f'Model: {engine.model_family}; Mode: infer; Input: {mode}; WAV duration: {info["duration_seconds"]:.3f} s', flush=True)
        print('-'*64, flush=True)
    started = time.perf_counter()
    with responses_path.open('w', encoding='utf-8') as writer:
        def record(result):
            row = {**result, 'transport': TRANSPORT, 'source_wav': str(wav_path)}
            writer.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            writer.flush()
            if display:
                _print_result(row)

        if mode == 'segment':
            segment_started = time.perf_counter()
            result = monitor.segment()
            if display:
                print('Segment scope: first complete window [0.000, 5.000) s only.', flush=True)
                _print_request('Segment', result['request'])
            record(result)
            report['segment'] = {'scope': 'first_complete_window_only', 'first_frame': 0,
                                 'end_frame': WINDOW_FRAMES, 'duration_seconds': 5.,
                                 'remaining_source_frames_not_sent': info['frame_count']-WINDOW_FRAMES,
                                 'wall_time_seconds': time.perf_counter()-segment_started, 'response': result}
        if mode in ('file', 'both'):
            if display:
                print('File scope: entire WAV; all complete five-second windows; no real-time pacing.', flush=True)
            report['file'] = monitor.file(record)
        if mode in ('stream', 'both'):
            if display:
                print('-'*64, flush=True)
                print(f'Stream scope: new session from frame 0; {"real-time" if realtime else "fast"} replay.', flush=True)
            report['stream'] = monitor.stream(round(SAMPLE_RATE*chunk_ms/1000), realtime, record, display)
    if mode == 'both':
        segment = report['file']['first_response']
        first = report['stream']['first_response']
        difference = max(abs(segment['class_probabilities'][name]-first['class_probabilities'][name])
                         for name in engine.class_names)
        labels_match = segment['predicted_class_index'] == first['predicted_class_index']
        report['first_window_comparison'] = {'same_source_range': [0, WINDOW_FRAMES],
                                             'labels_match': labels_match,
                                             'max_probability_difference': difference,
                                             'absolute_tolerance': 1e-7,
                                             'matches': bool(labels_match and difference <= 1e-7)}
    report['simulation_wall_time_seconds'] = time.perf_counter()-started
    report['inference_calls'] = int(report['segment'] is not None)+sum(
        report[kind]['emitted_windows'] for kind in ('file', 'stream') if report[kind])
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    if display:
        print('-'*64, flush=True)
        for kind in ('file', 'stream'):
            if report[kind]:
                summary = report[kind]
                print(f"{kind.title()} windows: {summary['emitted_windows']}; tail: {summary['trailing_frames']} frames "
                      f"({summary['trailing_duration_s']:.3f} s), not inferred.", flush=True)
        if report['first_window_comparison']:
            comparison = report['first_window_comparison']
            print(f"First file / stream window match: {comparison['matches']}; "
                  f"max probability difference={comparison['max_probability_difference']:.3g}", flush=True)
        if metadata.get('synthetic_only') is not True:
            print('Class scores are uncalibrated.', flush=True)
        print(f'Report: {report_path}', flush=True)
        print(f'Responses: {responses_path}', flush=True)
        print('-'*64, flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=FAMILIES, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--wav', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('infer', 'segment', 'stream', 'both'), default='infer',
                        help='infer: load saved weights and predict (default); segment/stream/both remain legacy input-mode aliases')
    parser.add_argument('--input-mode', choices=('file', 'stream', 'both', 'segment'),
                        help='file: all complete 5-second windows without pacing; stream: chunked WAV replay; both: file then stream (default); segment: legacy first-window-only demo')
    parser.add_argument('--chunk-ms', type=float, default=20.)
    parser.add_argument('--realtime', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'),
                        help='auto: use CUDA after a successful probe, otherwise CPU (default)')
    args = parser.parse_args(argv)
    if args.mode != 'infer' and args.input_mode is not None and args.input_mode != args.mode:
        parser.error('Legacy --mode conflicts with --input-mode; use --mode infer to choose the input mode separately.')
    input_mode = args.input_mode or ('both' if args.mode == 'infer' else args.mode)
    device_options = {'device': args.device} if args.device is not None else {}
    try:
        run_simulation(args.model, args.checkpoint, args.wav, args.output_dir,
                       mode=input_mode, chunk_ms=args.chunk_ms, realtime=args.realtime, threads=args.threads,
                       **device_options)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
