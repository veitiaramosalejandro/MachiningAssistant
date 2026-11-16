"""Export nonoverlapping PCM WAV segments with split and event provenance.

The editable split CSV assigns complete source groups before segmentation.
Original signals receive the professor-agreed ``good`` label; manual event
state and quality annotations are preserved separately.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import sys
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from Dataset.Preprocessing.dataset_splits import SPLIT_NAMES, read_split_csv, write_split_csv


SAMPLE_RATE = 44100
CHANNELS = 2
SAMPLE_WIDTH = 2
FRAME_BYTES = CHANNELS*SAMPLE_WIDTH
DEFAULT_DATA_DIR = PROJECT_ROOT/'Dataset'/'Data'
DEFAULT_SPLIT_CSV = Path(__file__).resolve().parent/'dataset_splits.csv'
DEFAULT_OUTPUT = DEFAULT_DATA_DIR/'segments_5s'
LABEL_ORIGIN = 'professor_convention_original_signal_is_good'
ALL_SPLITS = (*SPLIT_NAMES, 'exclude')
CSV_FIELDS = (
    'sample_id', 'base_sample_id', 'recording_id', 'source_recording_id', 'split',
    'wav_file', 'source_wav', 'start_frame', 'end_frame_exclusive',
    'start_seconds', 'end_seconds_exclusive', 'duration_seconds', 'frame_count',
    'is_full_segment', 'label', 'label_origin', 'machine_state', 'machine_state_status',
    'quality_score', 'quality_status', 'event_intervals',
)


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def _relative(path, root):
    return Path(os.path.relpath(path, root)).as_posix()


def _original_source(recording_id, metadata):
    original = metadata.get('SourceRecording')
    if original is None:
        return recording_id, 0.0
    folder = original.get('Folder') if isinstance(original, dict) else None
    if not isinstance(folder, str) or not folder.strip().strip('/\\'):
        raise ValueError(f'{recording_id}: SourceRecording.Folder is missing')
    source_id = folder.strip().replace('\\', '/').rstrip('/').split('/')[-1]
    origin = original.get('StartSeconds')
    if origin is not None:
        origin = float(origin)
        if not math.isfinite(origin):
            raise ValueError(f'{recording_id}: SourceRecording.StartSeconds must be finite')
    return source_id, origin


def _event_timeline(folder):
    path = folder/'events.json'
    document = _json(path) if path.is_file() else {}
    annotations = document.get('annotations', [])
    if not isinstance(annotations, list):
        raise ValueError(f'{path}: annotations must be a list')
    playback_path = folder/'playback.json'
    playback = _json(playback_path) if playback_path.is_file() else {}
    offset = float(playback.get('AudioOffsetMs', 0))
    if not math.isfinite(offset):
        raise ValueError(f'{folder}: current playback audio offset must be finite')
    if annotations:
        if document.get('time_reference') not in ('playback_seconds', 'wav_seconds'):
            raise ValueError(f'{path}: unsupported or missing event time_reference')
        if offset != 0:
            raise ValueError(f'{folder}: events require resolved time alignment; current playback AudioOffsetMs is {offset}, not zero')
    ordered = []
    for order, event in enumerate(annotations):
        timestamp = float(event['timestamp_seconds'])
        if not math.isfinite(timestamp):
            raise ValueError(f'{path}: annotation timestamp must be finite')
        ordered.append((timestamp, order, event))
    ordered.sort(key=lambda item: (item[0], item[1]))
    state = {'machine_state': None, 'quality_score': None,
             'machine_state_annotation_id': None, 'machine_state_annotation_source_seconds': None,
             'quality_annotation_id': None, 'quality_annotation_source_seconds': None}
    timeline = []
    for timestamp, _, event in ordered:
        # Each valid snapshot value is usable even when its change flag is false.
        machine_state = event.get('machine_state')
        quality = event.get('quality')
        if isinstance(machine_state, str) and machine_state.strip():
            state.update(machine_state=machine_state,
                         machine_state_annotation_id=event.get('id'),
                         machine_state_annotation_source_seconds=timestamp)
        if isinstance(quality, (int, float)) and not isinstance(quality, bool) and math.isfinite(quality):
            state.update(quality_score=quality, quality_annotation_id=event.get('id'),
                         quality_annotation_source_seconds=timestamp)
        snapshot = {**state, 'timestamp_seconds': timestamp, 'annotation_id': event.get('id'),
                    'event_changes': event.get('event_changes', {})}
        if timeline and timeline[-1]['timestamp_seconds'] == timestamp:
            timeline[-1] = snapshot
        else:
            timeline.append(snapshot)
    return timeline, offset, len(annotations), document.get('time_reference')


def _event_intervals(timeline, start, end, original_origin):
    """Represent [start,end) using preceding snapshots and in-segment changes."""
    if not timeline:
        return []
    timestamps = [event['timestamp_seconds'] for event in timeline]
    boundaries = [start] + [t for t in timestamps if start < t < end] + [end]
    result = []
    for left, right in zip(boundaries, boundaries[1:]):
        index = bisect.bisect_right(timestamps, left)-1
        event = timeline[index] if index >= 0 else {}
        result.append({
            'start_seconds': left-start,
            'end_seconds_exclusive': right-start,
            'duration_seconds': right-left,
            'source_start_seconds': left,
            'source_end_seconds_exclusive': right,
            'original_recording_start_seconds': original_origin+left if original_origin is not None else None,
            'original_recording_end_seconds_exclusive': original_origin+right if original_origin is not None else None,
            'machine_state': event.get('machine_state'),
            'quality_score': event.get('quality_score'),
            'annotation_id': event.get('annotation_id'),
            'annotation_source_seconds': event.get('timestamp_seconds'),
            'inherited_before_segment': index >= 0 and event['timestamp_seconds'] < start,
            'machine_state_annotation_id': event.get('machine_state_annotation_id'),
            'machine_state_annotation_source_seconds': event.get('machine_state_annotation_source_seconds'),
            'quality_annotation_id': event.get('quality_annotation_id'),
            'quality_annotation_source_seconds': event.get('quality_annotation_source_seconds'),
            'event_changes': event.get('event_changes', {}),
        })
    return result


def _single_annotation(intervals, field):
    values = [interval[field] for interval in intervals]
    known = {value for value in values if value is not None}
    if not known:
        return None, 'unannotated'
    if len(known) > 1:
        return None, 'mixed'
    if any(value is None for value in values):
        return None, 'partially_annotated'
    return next(iter(known)), 'annotated_constant'


def _write_wav(path, payload, frame_count):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(CHANNELS)
        output.setsampwidth(SAMPLE_WIDTH)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(payload)
    with wave.open(str(path), 'rb') as saved:
        if (saved.getframerate(), saved.getnchannels(), saved.getsampwidth(), saved.getnframes()) != (
            SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH, frame_count
        ):
            raise AssertionError(f'Unexpected saved WAV header: {path}')
        if saved.readframes(frame_count) != payload:
            raise AssertionError(f'Saved PCM bytes differ from source: {path}')


def _write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'event_intervals': json.dumps(row['event_intervals'], ensure_ascii=False, separators=(',', ':'))})


def build_dataset_segments(data_dir, split_csv, output_dir, segment_seconds=5.0):
    """Export every CSV row, keeping excluded recordings outside active splits."""
    data_dir = Path(data_dir).expanduser().resolve()
    split_csv = Path(split_csv).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not math.isfinite(segment_seconds) or segment_seconds <= 0:
        raise ValueError('segment_seconds must be finite and positive')
    segment_frames = round(segment_seconds*SAMPLE_RATE)
    if segment_frames < 1:
        raise ValueError('segment_seconds must span at least one PCM frame')
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f'Output directory is not empty: {output_dir}')
    assignments = read_split_csv(split_csv)
    groups = {split: sorted({r['source_recording_id'] for r in assignments if r['split'] == split}) for split in ALL_SPLITS}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if set(groups[a]) & set(groups[b]):
            raise ValueError(f'Source recording groups overlap between {a} and {b}')
    prepared = []
    for row in assignments:
        folder = data_dir/row['recording_id']
        metadata = _json(folder/'metadata.json')
        actual_source, original_origin = _original_source(row['recording_id'], metadata)
        if actual_source != row['source_recording_id']:
            raise ValueError(f"{row['recording_id']}: source_recording_id mismatch: CSV={row['source_recording_id']!r}, metadata={actual_source!r}")
        timeline, offset, annotation_count, time_reference = _event_timeline(folder)
        wav_path = folder/'audio_000000.wav'
        with wave.open(str(wav_path), 'rb') as source:
            if (source.getframerate(), source.getnchannels(), source.getsampwidth(), source.getcomptype()) != (
                SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH, 'NONE'
            ):
                raise ValueError(f'{wav_path}: expected 44100 Hz stereo PCM16')
            frames = source.getnframes()
        prepared.append({**row, 'wav_path': wav_path, 'frames': frames, 'timeline': timeline,
                         'original_origin': original_origin, 'playback_audio_offset_ms': offset,
                         'annotations_count': annotation_count, 'event_time_reference': time_reference})

    output_dir.mkdir(parents=True, exist_ok=True)
    segments, tails, records = [], [], []
    sample_ids = set()
    for record in prepared:
        recording_id, split = record['recording_id'], record['split']
        full_count, remainder = divmod(record['frames'], segment_frames)
        local_rows = []
        cursor = 0
        with wave.open(str(record['wav_path']), 'rb') as source:
            for index in range(full_count+int(remainder > 0)):
                full = index < full_count
                count = segment_frames if full else remainder
                end = cursor+count
                payload = source.readframes(count)
                if len(payload) != count*FRAME_BYTES:
                    raise ValueError(f"{record['wav_path']}: source PCM payload is shorter than its declared frames")
                relative_file = (Path(split)/recording_id/f'segment_{index:04d}.wav' if full else
                                 Path('tails')/split/recording_id/'tail.wav')
                sample_id = relative_file.with_suffix('').as_posix()
                if sample_id in sample_ids:
                    raise AssertionError(f'Duplicate sample_id: {sample_id}')
                sample_ids.add(sample_id)
                intervals = _event_intervals(record['timeline'], cursor/SAMPLE_RATE, end/SAMPLE_RATE,
                                            record['original_origin'])
                state, state_status = _single_annotation(intervals, 'machine_state')
                quality, quality_status = _single_annotation(intervals, 'quality_score')
                result = {
                    'sample_id': sample_id, 'base_sample_id': sample_id,
                    'recording_id': recording_id, 'source_recording_id': record['source_recording_id'],
                    'split': split, 'wav_file': relative_file.as_posix(),
                    'source_wav': _relative(record['wav_path'], output_dir),
                    'start_frame': cursor, 'end_frame_exclusive': end,
                    'start_seconds': cursor/SAMPLE_RATE, 'end_seconds_exclusive': end/SAMPLE_RATE,
                    'duration_seconds': count/SAMPLE_RATE, 'frame_count': count,
                    'is_full_segment': full, 'label': 'good', 'label_origin': LABEL_ORIGIN,
                    'machine_state': state, 'machine_state_status': state_status,
                    'quality_score': quality, 'quality_status': quality_status,
                    'event_intervals': intervals,
                }
                _write_wav(output_dir/relative_file, payload, count)
                (segments if full else tails).append(result)
                local_rows.append(result)
                cursor = end
            if source.readframes(1):
                raise AssertionError(f"Unexpected unread PCM after declared source frames: {record['wav_path']}")
        if cursor != record['frames'] or sum(row['frame_count'] for row in local_rows) != record['frames']:
            raise AssertionError(f'Frame coverage mismatch for {recording_id}')
        if any(a['end_frame_exclusive'] != b['start_frame'] for a, b in zip(local_rows, local_rows[1:])):
            raise AssertionError(f'Gap or overlap in {recording_id}')
        records.append({
            'recording_id': recording_id, 'source_recording_id': record['source_recording_id'], 'split': split,
            'source_wav': _relative(record['wav_path'], output_dir),
            'source_frame_count': record['frames'], 'source_duration_s': record['frames']/SAMPLE_RATE,
            'full_segments': full_count, 'full_frame_count': full_count*segment_frames,
            'full_duration_s': full_count*segment_frames/SAMPLE_RATE,
            'tail_count': int(remainder > 0), 'tail_frame_count': remainder, 'tail_duration_s': remainder/SAMPLE_RATE,
            'covered_frame_count': cursor, 'covers_all_frames_once': True,
            'event_annotations_count': record['annotations_count'], 'event_time_reference': record['event_time_reference'],
            'playback_audio_offset_ms': record['playback_audio_offset_ms'],
            'original_recording_start_seconds': record['original_origin'],
            'machine_state_status_counts': dict(Counter(row['machine_state_status'] for row in local_rows)),
            'quality_status_counts': dict(Counter(row['quality_status'] for row in local_rows)),
        })
        print(f'{recording_id}: {full_count} full segments, {remainder} tail frames ({split})', flush=True)

    _write_csv(output_dir/'segments.csv', segments)
    _write_json(output_dir/'segments.json', segments)
    for split in ALL_SPLITS:
        name = 'excluded.csv' if split == 'exclude' else f'{split}.csv'
        _write_csv(output_dir/name, [row for row in segments if row['split'] == split])
    _write_csv(output_dir/'tails.csv', tails)
    _write_json(output_dir/'tails.json', tails)
    write_split_csv(output_dir/'dataset_splits_used.csv', assignments)
    split_counts = {split: sum(row['split'] == split for row in segments) for split in ALL_SPLITS}
    tail_counts = {split: sum(row['split'] == split for row in tails) for split in ALL_SPLITS}
    source_frames = sum(record['source_frame_count'] for record in records)
    full_frames = sum(row['frame_count'] for row in segments)
    tail_frames = sum(row['frame_count'] for row in tails)
    if source_frames != full_frames+tail_frames:
        raise AssertionError('Global frame conservation failed')
    summary = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'data_dir': str(data_dir), 'split_csv': str(split_csv), 'output_dir': str(output_dir),
        'sample_rate_hz': SAMPLE_RATE, 'channels': CHANNELS, 'bit_depth': 16,
        'requested_segment_seconds': segment_seconds, 'segment_seconds': segment_frames/SAMPLE_RATE,
        'segment_frames': segment_frames, 'segment_hop_frames': segment_frames,
        'recording_count': len(records), 'full_segment_count': len(segments), 'tail_segment_count': len(tails),
        'split_counts': split_counts, 'tail_counts': tail_counts, 'source_groups': groups,
        'source_group_counts': {split: len(value) for split, value in groups.items()},
        'source_frame_count': source_frames, 'full_frame_count': full_frames, 'tail_frame_count': tail_frames,
        'source_duration_s': source_frames/SAMPLE_RATE, 'full_duration_s': full_frames/SAMPLE_RATE,
        'tail_duration_s': tail_frames/SAMPLE_RATE,
        'label_counts': {'good': len(segments)}, 'tail_label_counts': {'good': len(tails)},
        'label_origin': LABEL_ORIGIN,
        'label_definition': 'Original undisturbed signals are labelled good by the professor agreement; this label is not derived from event quality=3 or predicted by a model.',
        'only_original_good': True, 'synthetic_variants_generated': False,
        'synthetic_classes_not_generated': ['sufficient', 'bad', 'very_bad'],
        'event_interval_time_definition': {
            'start_seconds/end_seconds_exclusive': 'Relative to the exported segment or tail.',
            'source_start_seconds/source_end_seconds_exclusive': 'Relative to the input recording WAV.',
            'original_recording_start_seconds/original_recording_end_seconds_exclusive': 'Input WAV time plus metadata.SourceRecording.StartSeconds, when known.',
            'boundary': '[start,end): an event exactly at the right boundary belongs to the next interval.',
        },
        'annotation_policy': 'Inherit the latest valid snapshot at or before a slice boundary. False change flags do not discard valid snapshot values. Missing values remain null; mixed slices have no single state/quality value.',
        'audio_alignment_policy': 'Use current playback offset only; nonempty events with nonzero current offset are rejected. Historical SourceRecording.AudioOffsetMs is not reapplied.',
        'tail_policy': 'Write each short tail separately with its true length; no zero padding or cross-recording concatenation.',
        'verification': {'active_source_groups_disjoint': True, 'unique_sample_ids': True,
                         'covers_all_frames_once': True, 'all_saved_pcm_bytes_match_source': True,
                         'resampled': False, 'normalized': False, 'padded': False},
        'recordings': records,
    }
    _write_json(output_dir/'dataset_summary.json', summary)
    readme = f'''Original {segment_frames/SAMPLE_RATE:g}-second signal samples
================================

This export contains {len(segments)} complete segments and {len(tails)} separate short tails from {len(records)} input WAVs.
The realized segment length is {segment_frames} frames ({segment_frames/SAMPLE_RATE:g} seconds) at 44100 Hz, stereo PCM16.
Raw PCM payload bytes are preserved; each exported WAV has its own container header.

Directories and tables
---------------------
train/, val/, test/, exclude/: complete segments, grouped by input recording folder.
tails/<split>/<recording_id>/tail.wav: each input's incomplete final segment.
segments.csv / segments.json: all complete segments, including excluded recordings.
train.csv / val.csv / test.csv / excluded.csv: complete-segment subsets.
tails.csv / tails.json: short-tail records, separate from full-length training samples.
dataset_splits_used.csv: the exact input split assignments.
dataset_summary.json: counts, frame coverage, source groups, and provenance.

The CSV was applied before slicing. Training, validation, and test source-recording groups do not overlap.
The recording_id remains in every file path so different source clips cannot collide at local time zero.
base_sample_id currently equals sample_id and identifies the original sample for later variants.

Labels and events
-----------------
Every exported original waveform has label=good under the professor's original-signal agreement.
This is independent of machine_state and quality_score from events.json. No synthetic sufficient,
bad, or very_bad variants have been generated, and no model has been trained by this script.
Missing state or quality is null in JSON and blank in CSV, never an inferred zero.
Mixed-state segments have machine_state_status=mixed and no single machine_state.
If quality stays 3 through a state change, quality_status remains annotated_constant.

event_intervals is a JSON array in the JSON files and a serialized JSON array in CSV.
Its start_seconds/end_seconds_exclusive are segment-relative. source_* times are relative
to the input WAV. original_recording_* times additionally include SourceRecording.StartSeconds.
Intervals are [start,end), and the state valid before a segment starts is inherited.
Current playback offsets are checked; historical SourceRecording.AudioOffsetMs is not applied again.

Reproduce from the project root
-------------------------------
python Dataset/Preprocessing/build_segments.py --output /path/to/new_output
Optional flags: --data-dir, --split-csv, --segment-seconds.
A nonempty output directory is rejected. Original recordings and the editable CSV remain unchanged.
'''
    (output_dir/'README.txt').write_text(readme, encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--split-csv', type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument('--segment-seconds', type=float, default=5.0)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = build_dataset_segments(args.data_dir, args.split_csv, args.output, args.segment_seconds)
    print(json.dumps({key: summary[key] for key in ('output_dir', 'recording_count', 'full_segment_count',
                                                    'tail_segment_count', 'split_counts', 'tail_counts',
                                                    'full_duration_s', 'tail_duration_s')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
