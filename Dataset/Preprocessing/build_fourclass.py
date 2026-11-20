"""Build a four-class sine-injection dataset from complete original segments.

The reference RMS is fitted only on original training segments. Local random
parameters use a stable source-window identity and do not depend on traversal
order or split names. No model training is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from Dataset.Preprocessing.dataset_splits import SPLIT_NAMES, read_split_csv, write_split_csv


DEFAULT_SOURCE = PROJECT_ROOT/'Dataset'/'Data'/'segments_5s'
DEFAULT_OUTPUT = PROJECT_ROOT/'Dataset'/'Data'/'fourclass_5s'
DEFAULT_CONFIG = Path(__file__).resolve().parent/'fourclass_config.json'
LABELS =  ('excellent', 'good', 'sufficient', 'bad', 'very_bad', 'broken')
ALL_SPLITS = (*SPLIT_NAMES, 'exclude')
FIELDS = (
    'sample_id', 'base_sample_id', 'recording_id', 'source_recording_id', 'split',
    'label', 'label_index', 'label_origin', 'synthetic_only', 'is_synthetic_signal',
    'wav_file', 'base_wav', 'source_wav',
    'start_frame', 'end_frame_exclusive', 'start_seconds', 'end_seconds_exclusive',
    'duration_seconds', 'frame_count', 'sample_rate_hz', 'channels', 'bit_depth',
    'generation_frequency_hz', 'injection_channel', 'peak_amplitude_ratio',
    'peak_amplitude_pcm', 'phase_radians', 'jitter_multiplier', 'reference_rms_pcm',
    'right_changed_frame_count', 'left_changed_frame_count',
    'original_label', 'original_label_origin', 'original_machine_state',
    'original_machine_state_status', 'original_quality_score', 'original_quality_status',
    'original_event_intervals',
)


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def _relative(path, output):
    return Path(os.path.relpath(path, output)).as_posix()


def _validated_config(config):
    result = dict(config)
    if result.get('sample_rate') != 44100 or result.get('channels') != 2:
        raise ValueError('Expected 44100 Hz and two channels')
    if result.get('injection_channel') != 1:
        raise ValueError('This dataset injects only the right channel (index 1)')
    if tuple(result.get('labels', ())) != LABELS:
        raise ValueError('Expected labels good, sufficient, bad, very_bad in that order')
    duration = float(result['segment_seconds'])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('segment_seconds must be finite and positive')
    frames = round(duration*result['sample_rate'])
    if frames < 1:
        raise ValueError('segment_seconds must include at least one PCM frame')
    frequency = float(result['sine_frequency_hz'])
    if not math.isfinite(frequency) or not 0 < frequency < result['sample_rate']/2:
        raise ValueError('Sine frequency must be positive and below Nyquist')
    ratios = [float(value) for value in result['peak_amplitude_ratios']]
    if len(ratios) != 4 or ratios[0] != 0 or any(not math.isfinite(value) for value in ratios):
        raise ValueError('Four finite amplitude ratios are required, starting with zero for good')
    if any(b <= a for a, b in zip(ratios, ratios[1:])):
        raise ValueError('Amplitude ratios must increase strictly from good to very_bad')
    jitter = float(result['amplitude_jitter_fraction'])
    if not math.isfinite(jitter) or not 0 <= jitter < 1:
        raise ValueError('amplitude_jitter_fraction must be between zero and one')
    if result.get('reference_statistic') != 'median_training_segment_rms':
        raise ValueError('Supported reference statistic: median_training_segment_rms')
    if result.get('phase_policy') != 'shared_random_phase_per_base_segment':
        raise ValueError('All amplitude variants must share one phase per base segment')
    if result.get('jitter_policy') != 'shared_amplitude_multiplier_per_base_segment':
        raise ValueError('All amplitude variants must share one amplitude multiplier per base segment')
    seed = int(result['seed'])
    if seed < 0:
        raise ValueError('seed must be nonnegative')
    result.update(segment_seconds=duration, segment_frames=frames,
                  sine_frequency_hz=frequency, peak_amplitude_ratios=ratios,
                  amplitude_jitter_fraction=jitter, seed=seed)
    return result


def _read_pcm(path, frames, rate):
    with wave.open(str(path), 'rb') as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype(), wav.getnframes()) != (
            rate, 2, 2, 'NONE', frames
        ):
            raise ValueError(f'{path}: expected a complete {frames}-frame stereo PCM16 segment at {rate} Hz')
        payload = wav.readframes(frames)
    if len(payload) != frames*4:
        raise ValueError(f'{path}: incomplete PCM payload')
    return payload, np.frombuffer(payload, dtype='<i2').reshape(frames, 2)


def _random_parameters(base, config):
    identity = json.dumps([base['recording_id'], int(base['start_frame']), int(base['end_frame_exclusive'])],
                          ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    rng = np.random.default_rng(np.random.SeedSequence([config['seed'], *identity]))
    phase = float(rng.uniform(0., 2*np.pi))
    fraction = config['amplitude_jitter_fraction']
    multiplier = float(rng.uniform(1.-fraction, 1.+fraction))
    return phase, multiplier


def _injected_right(original, peak_amplitude, sine, base_id, label):
    values = np.rint(original.astype(np.float64)+peak_amplitude*sine)
    if values.min() < -32768 or values.max() > 32767:
        raise ValueError(f'PCM16 overflow for {base_id} / {label}; amplitudes were not clipped or adjusted')
    return values.astype('<i2')


def _write_and_verify(path, payload, frames, rate):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(2); wav.setsampwidth(2); wav.setframerate(rate)
        wav.writeframes(payload)
    saved_payload, saved = _read_pcm(path, frames, rate)
    if saved_payload != payload:
        raise AssertionError(f'Saved PCM bytes differ from generated signal: {path}')
    return saved_payload, saved


def _write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'original_event_intervals': json.dumps(
                row['original_event_intervals'], ensure_ascii=False, separators=(',', ':'))})


def build_fourclass_dataset(source_dir, output_dir, config: dict):
    """Write four real WAV variants for each complete original-good segment."""
    source_dir = Path(source_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f'Output directory is not empty: {output_dir}')
    parameters = _validated_config(config)
    rate, frames = parameters['sample_rate'], parameters['segment_frames']
    assignments = read_split_csv(source_dir/'dataset_splits_used.csv')
    assigned = {row['recording_id']: row for row in assignments}
    groups = {split: sorted({row['source_recording_id'] for row in assignments if row['split'] == split})
              for split in ALL_SPLITS}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if set(groups[a]) & set(groups[b]):
            raise ValueError(f'Source groups overlap between {a} and {b}')
    bases = _json(source_dir/'segments.json')
    if not isinstance(bases, list) or not bases:
        raise ValueError('segments.json must contain complete original base segments')
    seen_bases, seen_files = set(), set()
    prepared, reference_samples = [], []
    right_min, right_max = 32767, -32768
    for base in bases:
        base_id = base['base_sample_id']
        if base_id in seen_bases:
            raise ValueError(f'Duplicate base_sample_id: {base_id}')
        seen_bases.add(base_id)
        assignment = assigned.get(base['recording_id'])
        if assignment is None or any(base[key] != assignment[key] for key in ('source_recording_id', 'split')):
            raise ValueError(f'{base_id}: source/split does not match dataset_splits_used.csv')
        if base.get('label') != 'good' or base.get('is_full_segment') is not True:
            raise ValueError(f'{base_id}: input must be an unmodified complete good segment')
        if base['frame_count'] != frames or base['end_frame_exclusive']-base['start_frame'] != frames:
            raise ValueError(f'{base_id}: base length differs from the configured segment length')
        if not math.isclose(base['duration_seconds'], frames/rate, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f'{base_id}: inconsistent source duration')
        path = (source_dir/base['wav_file']).resolve()
        filename_key = (base['split'], base['recording_id'], path.name)
        if filename_key in seen_files:
            raise ValueError(f'Duplicate output base filename for {base_id}')
        seen_files.add(filename_key)
        _, pcm = _read_pcm(path, frames, rate)
        right_min = min(right_min, int(pcm[:, 1].min()))
        right_max = max(right_max, int(pcm[:, 1].max()))
        if base['split'] == 'train':
            rms = float(np.sqrt(np.mean(pcm[:, 1].astype(np.float64)**2)))
            reference_samples.append({'base_sample_id': base_id, 'recording_id': base['recording_id'],
                                      'source_recording_id': base['source_recording_id'],
                                      'start_frame': base['start_frame'], 'end_frame_exclusive': base['end_frame_exclusive'],
                                      'rms_pcm': rms})
        prepared.append({'base': base, 'base_path': path,
                         'original_path': (source_dir/base['source_wav']).resolve()})
    if not reference_samples:
        raise ValueError('No original training segments are available to fit the reference RMS')
    reference_rms = float(np.median([row['rms_pcm'] for row in reference_samples]))
    if not math.isfinite(reference_rms) or reference_rms <= 0:
        raise ValueError('The training reference RMS must be positive for distinct injected classes')
    ratios = parameters['peak_amplitude_ratios']
    max_peak = max(ratios)*reference_rms*(1.+parameters['amplitude_jitter_fraction'])
    conservative_safe = right_min-max_peak >= -32768 and right_max+max_peak <= 32767
    time_axis = np.arange(frames, dtype=np.float64)/rate
    if not conservative_safe:
        # Check actual quantized outputs before creating a partial dataset; never reduce amplitudes.
        for item in prepared:
            _, pcm = _read_pcm(item['base_path'], frames, rate)
            phase, jitter = _random_parameters(item['base'], parameters)
            sine = np.sin(2*np.pi*parameters['sine_frequency_hz']*time_axis+phase)
            for label, ratio in zip(LABELS[1:], ratios[1:]):
                _injected_right(pcm[:, 1], ratio*reference_rms*jitter, sine,
                                item['base']['base_sample_id'], label)

    output_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    seen_sample_ids = set()
    changed_signals = 0
    for number, item in enumerate(prepared, start=1):
        base = item['base']
        original_payload, original = _read_pcm(item['base_path'], frames, rate)
        phase, jitter = _random_parameters(base, parameters)
        sine = np.sin(2*np.pi*parameters['sine_frequency_hz']*time_axis+phase)
        for label_index, (label, ratio) in enumerate(zip(LABELS, ratios)):
            amplitude = ratio*reference_rms*jitter
            if label == 'good':
                payload = original_payload
                right_changed = 0
            else:
                signal = original.copy()
                signal[:, 1] = _injected_right(original[:, 1], amplitude, sine, base['base_sample_id'], label)
                right_changed = int(np.count_nonzero(signal[:, 1] != original[:, 1]))
                if right_changed == 0:
                    raise ValueError(f"{base['base_sample_id']} / {label}: injection rounded to the original signal")
                payload = signal.astype('<i2', copy=False).tobytes()
                changed_signals += 1
            relative = Path(base['split'])/label/base['recording_id']/item['base_path'].name
            sample_id = relative.with_suffix('').as_posix()
            if sample_id in seen_sample_ids:
                raise AssertionError(f'Duplicate generated sample_id: {sample_id}')
            seen_sample_ids.add(sample_id)
            saved_payload, saved = _write_and_verify(output_dir/relative, payload, frames, rate)
            if not np.array_equal(saved[:, 0], original[:, 0]):
                raise AssertionError(f'Left channel changed for {sample_id}')
            if label == 'good' and saved_payload != original_payload:
                raise AssertionError(f'Good PCM payload changed for {sample_id}')
            samples.append({
                'sample_id': sample_id, 'base_sample_id': base['base_sample_id'],
                'recording_id': base['recording_id'], 'source_recording_id': base['source_recording_id'],
                'split': base['split'], 'label': label, 'label_index': label_index,
                'label_origin': (base.get('label_origin', 'professor_convention_original_signal_is_good')
                                 if label == 'good' else 'synthetic_sine_peak_amplitude_tier'),
                'synthetic_only': True, 'is_synthetic_signal': label != 'good',
                'wav_file': relative.as_posix(), 'base_wav': _relative(item['base_path'], output_dir),
                'source_wav': _relative(item['original_path'], output_dir),
                'start_frame': base['start_frame'], 'end_frame_exclusive': base['end_frame_exclusive'],
                'start_seconds': base['start_seconds'], 'end_seconds_exclusive': base['end_seconds_exclusive'],
                'duration_seconds': frames/rate, 'frame_count': frames,
                'sample_rate_hz': rate, 'channels': 2, 'bit_depth': 16,
                'generation_frequency_hz': parameters['sine_frequency_hz'], 'injection_channel': 1,
                'peak_amplitude_ratio': ratio, 'peak_amplitude_pcm': amplitude,
                'phase_radians': phase, 'jitter_multiplier': jitter, 'reference_rms_pcm': reference_rms,
                'right_changed_frame_count': right_changed, 'left_changed_frame_count': 0,
                'original_label': base['label'], 'original_label_origin': base.get('label_origin'),
                'original_machine_state': base.get('machine_state'),
                'original_machine_state_status': base.get('machine_state_status'),
                'original_quality_score': base.get('quality_score'),
                'original_quality_status': base.get('quality_status'),
                'original_event_intervals': base.get('event_intervals', []),
            })
        if number % 50 == 0 or number == len(prepared):
            print(f'Generated {number}/{len(prepared)} base groups ({len(samples)} WAV files)', flush=True)
    if len(samples) != len(bases)*4:
        raise AssertionError('Each base must produce four variants')
    by_base = {}
    by_source = {}
    for row in samples:
        by_base.setdefault(row['base_sample_id'], set()).add(row['split'])
        if row['split'] in SPLIT_NAMES:
            by_source.setdefault(row['source_recording_id'], set()).add(row['split'])
    if any(len(splits) != 1 for splits in by_base.values()) or any(len(splits) != 1 for splits in by_source.values()):
        raise AssertionError('Variants or original sources cross active splits')
    _write_csv(output_dir/'samples.csv', samples)
    _write_json(output_dir/'samples.json', samples)
    for split in ALL_SPLITS:
        _write_csv(output_dir/('excluded.csv' if split == 'exclude' else f'{split}.csv'),
                   [row for row in samples if row['split'] == split])
    _write_json(output_dir/'labels.json', {str(index): name for index, name in enumerate(LABELS)})
    write_split_csv(output_dir/'base_dataset_splits_used.csv', assignments)
    reference_samples.sort(key=lambda row: row['base_sample_id'])
    resolved = {
        **parameters, 'reference_rms_pcm': reference_rms, 'reference_sample_count': len(reference_samples),
        'reference_split': 'train', 'reference_channel': 1,
        'reference_definition': 'Median of sqrt(mean(raw_R_PCM^2)) over each original training base segment; no mean subtraction or standardization.',
        'reference_source_segments': _relative(source_dir/'segments.json', output_dir),
        'reference_split_snapshot': _relative(source_dir/'dataset_splits_used.csv', output_dir),
        'reference_training_samples': reference_samples,
        'nominal_peak_amplitude_pcm_by_label': {label: ratio*reference_rms for label, ratio in zip(LABELS, ratios)},
        'random_identity_fields': ['recording_id', 'start_frame', 'end_frame_exclusive'],
        'random_derivation': 'numpy.default_rng(SeedSequence([seed, *UTF8(JSON([recording_id,start_frame,end_frame_exclusive], ensure_ascii=False, separators=(comma,colon)))])); draw uniform phase then uniform shared jitter.',
        'quantization': 'Right output = np.rint(original_R + peak_amplitude_pcm * sin(2*pi*frequency*t + phase)); t starts at zero within the base segment.',
        'overflow_policy': 'Raise on PCM16 overflow; never clip or adjust amplitude using validation/test signals.',
        'observed_original_right_min_pcm': right_min, 'observed_original_right_max_pcm': right_max,
        'observed_original_right_abs_max_pcm': max(abs(right_min), abs(right_max)),
        'maximum_possible_peak_amplitude_pcm': max_peak,
        'conservative_global_bound_safe': bool(conservative_safe),
        'synthetic_only': True,
    }
    _write_json(output_dir/'resolved_config.json', resolved)
    split_counts = {split: sum(row['split'] == split for row in samples) for split in ALL_SPLITS}
    label_counts = {label: sum(row['label'] == label for row in samples) for label in LABELS}
    split_label_counts = {split: {label: sum(row['split'] == split and row['label'] == label for row in samples)
                                  for label in LABELS} for split in ALL_SPLITS}
    base_summary_file = source_dir/'dataset_summary.json'
    base_summary = _json(base_summary_file) if base_summary_file.is_file() else {}
    summary = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_dir': str(source_dir), 'output_dir': str(output_dir),
        'sample_rate_hz': rate, 'channels': 2, 'bit_depth': 16,
        'segment_seconds': frames/rate, 'segment_frames': frames,
        'base_sample_count': len(bases), 'sample_count': len(samples),
        'original_good_count': len(bases), 'modified_signal_count': changed_signals,
        'split_counts': split_counts, 'label_counts': label_counts, 'split_label_counts': split_label_counts,
        'base_split_counts': {split: sum(row['split'] == split for row in bases) for split in ALL_SPLITS},
        'assignment_recording_count': len(assignments),
        'recordings_with_full_segments': len({row['recording_id'] for row in bases}),
        'source_groups': groups, 'source_group_counts': {split: len(value) for split, value in groups.items()},
        'reference_rms_pcm': reference_rms, 'reference_sample_count': len(reference_samples),
        'nominal_peak_amplitude_pcm_by_label': resolved['nominal_peak_amplitude_pcm_by_label'],
        'excluded_tail_count': base_summary.get('tail_segment_count'),
        'tails_generated': False, 'source_signal_duration_s': len(bases)*frames/rate,
        'generated_signal_duration_s': len(samples)*frames/rate,
        'synthetic_only': True,
        'label_semantics': 'good retains the original waveform; sufficient/bad/very_bad are controlled synthetic sine-amplitude tiers, not observed machining-quality or fault grades.',
        'original_annotation_policy': 'Original machine state, quality score, and event intervals are retained under original_* fields; quality=3 is not a four-class severity label.',
        'parameter_status': parameters.get('parameter_status'),
        'verification': {'active_source_groups_disjoint': True, 'variants_stay_in_base_split': True,
                         'unique_base_sample_ids': True, 'unique_generated_sample_ids': True,
                         'good_pcm_identical_to_base': True, 'left_channel_identical_for_all_variants': True,
                         'every_saved_pcm_matches_generated_values': True, 'pcm16_overflow_count': 0,
                         'automatic_clipping_applied': False, 'reference_fitted_only_on_train': True},
    }
    _write_json(output_dir/'dataset_summary.json', summary)
    text = f'''Four-class synthetic signal dataset
===================================

{len(bases)} complete original base segments produce {len(samples)} real WAV files.
Format: {rate} Hz, stereo PCM16, {frames} frames ({frames/rate:g} seconds) per file.
No partial tails are included. Original recordings and base segments are unchanged.

Labels: 0=good, 1=sufficient, 2=bad, 3=very_bad.
good is copied without changing PCM bytes. The other three labels indicate artificial
sine-amplitude tiers, not independently measured machining quality or physical faults.
The whole four-class benchmark is marked synthetic_only=true; is_synthetic_signal=false
identifies the unchanged good baseline. Original event annotations remain in original_* fields.

Generation
----------
Only right channel index 1 receives a {parameters['sine_frequency_hz']:g} Hz sinusoid.
Reference RMS = {reference_rms:.15g} PCM, fitted as the median of per-segment raw right-channel
RMS over {len(reference_samples)} training base segments. Validation and test signals do not
enter this fit. Peak ratios: {ratios}. Shared amplitude jitter: +/-{parameters['amplitude_jitter_fraction']*100:g}%.
Each base's three injected variants use the same random phase and jitter multiplier;
only the amplitude ratio changes. Seed {parameters['seed']} is combined with UTF-8 bytes of
[recording_id,start_frame,end_frame_exclusive] using NumPy SeedSequence, independent of
split names, Python process hashes, and traversal order.
Right samples are rounded with np.rint; overflow raises an error rather than clipping.
All left-channel PCM bytes are preserved. No model was trained by this builder.

Files
-----
<split>/<label>/<recording_id>/<base-segment-filename>.wav: generated signals.
samples.csv/json: all generated samples; train.csv, val.csv, test.csv, excluded.csv: subsets.
labels.json: numeric class mapping. resolved_config.json: exact parameters and training reference IDs.
base_dataset_splits_used.csv: the source build's split snapshot, inherited before augmentation.
dataset_summary.json: actual counts and verification results.
base_sample_id groups variants from one original segment; source_recording_id groups original
acquisitions. Neither grouping crosses training/validation/test splits.
Row start/end frames and seconds refer to the original input recording window; each new WAV
starts at local sample zero. base_wav and source_wav are relative to this output root.

Reproduce from the project root
-------------------------------
python Dataset/Preprocessing/build_fourclass.py --output /path/to/new_output
Optional: --source, --config. A nonempty output directory is rejected.
'''
    (output_dir/'README.txt').write_text(text, encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    summary = build_fourclass_dataset(args.source, args.output, _json(args.config))
    print(json.dumps({key: summary[key] for key in ('output_dir', 'base_sample_count', 'sample_count',
                                                    'split_counts', 'label_counts', 'reference_rms_pcm',
                                                    'reference_sample_count')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
