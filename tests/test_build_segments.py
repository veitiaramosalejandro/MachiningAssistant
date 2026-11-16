"""Verify lossless PCM segmentation, source isolation, and event alignment."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import wave

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from Dataset.Preprocessing.build_segments import build_dataset_segments
from Dataset.Preprocessing.dataset_splits import write_split_csv


FS = 44100


def assignment(recording, split, source=None):
    return {'recording_id': recording, 'source_recording_id': source or recording, 'split': split}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def read_pcm(path):
    with wave.open(str(path), 'rb') as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) == (FS, 2, 2, 'NONE')
        frames = wav.getnframes()
        payload = wav.readframes(frames)
    assert len(payload) == frames*4
    return frames, payload


def write_recording(root, name, frames, *, source=None, source_start=0, events=None, playback_ms=None, seed=0):
    folder = root/name
    folder.mkdir(parents=True)
    index = np.arange(frames, dtype=np.int64)
    left = ((index*37 + seed) % 65536 - 32768).astype('<i2')
    right = ((index*(-71) + 12345 + seed) % 65536 - 32768).astype('<i2')
    pcm = np.column_stack((left, right)).astype('<i2')
    payload = pcm.tobytes()
    with wave.open(str(folder/'audio_000000.wav'), 'wb') as wav:
        wav.setnchannels(2); wav.setsampwidth(2); wav.setframerate(FS)
        wav.writeframes(payload)
    metadata = {'DurationSeconds': frames/FS,
                'Context': {'Machine': 'fixture-machine', 'Operation': 'fixture-operation'}}
    if source is not None:
        metadata['SourceRecording'] = {'Folder': 'D:\\raw\\' + source,
                                       'StartSeconds': source_start,
                                       'EndSeconds': source_start + frames/FS,
                                       'AudioOffsetMs': 1400}
    (folder/'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    if events is not None:
        (folder/'events.json').write_text(json.dumps({
            'schema_version': 1, 'time_reference': 'playback_seconds', 'annotations': events,
        }), encoding='utf-8')
    if playback_ms is not None:
        (folder/'playback.json').write_text(json.dumps({'AudioOffsetMs': playback_ms}), encoding='utf-8')
    return payload


def event(identifier, seconds, machine_state, quality):
    return {'id': identifier, 'timestamp_seconds': seconds, 'machine_state': machine_state,
            'quality': quality, 'event_changes': {'machine_state': True, 'quality': True}}


def write_inputs(tmp_path, specifications):
    data = tmp_path/'data'
    data.mkdir()
    rows = []
    payloads = {}
    for name, split, frames in specifications:
        payloads[name] = write_recording(data, name, frames)
        rows.append(assignment(name, split))
    splits = tmp_path/'dataset_splits.csv'
    write_split_csv(splits, rows)
    return data, splits, rows, payloads


def test_real_five_second_segments_and_tails_restore_every_pcm_byte(tmp_path):
    five_seconds = 220500
    specs = [('train_recording', 'train', five_seconds*2 + 17),
             ('val_recording', 'val', five_seconds + 100),
             ('test_recording', 'test', five_seconds),
             ('excluded_recording', 'exclude', five_seconds + 11)]
    data, splits, split_rows, payloads = write_inputs(tmp_path, specs)
    output = tmp_path/'segmented'
    summary = build_dataset_segments(data, splits, output, segment_seconds=5.0)
    full = read_json(output/'segments.json')
    tails = read_json(output/'tails.json')
    assert len(full) == 5
    assert len(tails) == 3
    assert len({row['sample_id'] for row in full}) == len(full)
    assert all(row['base_sample_id'] for row in full)
    assert read_json(output/'dataset_summary.json') == summary
    assert read_csv(output/'dataset_splits_used.csv') == split_rows

    for row in full:
        assert row['is_full_segment'] is True
        assert row['frame_count'] == five_seconds
        assert row['duration_seconds'] == 5.0
        assert row['end_frame_exclusive'] - row['start_frame'] == five_seconds
        assert row['label'] == 'good'
        assert row['machine_state'] is None and row['quality_score'] is None
        assert row['machine_state_status'] == row['quality_status'] == 'unannotated'
        path = Path(row['wav_file'])
        assert not path.is_absolute()
        assert path.parts[:2] == (row['split'], row['recording_id'])
    for row in tails:
        assert row['is_full_segment'] is False
        assert 0 < row['frame_count'] < five_seconds
        assert Path(row['wav_file']).parts[:3] == ('tails', row['split'], row['recording_id'])

    for name, split, frame_count in specs:
        chunks = sorted([row for row in [*full, *tails] if row['recording_id'] == name],
                        key=lambda row: row['start_frame'])
        cursor = 0
        reconstructed = []
        for row in chunks:
            assert row['source_recording_id'] == name and row['split'] == split
            assert row['start_frame'] == cursor
            frames, payload = read_pcm(output/row['wav_file'])
            assert frames == row['frame_count']
            cursor += frames
            assert row['end_frame_exclusive'] == cursor
            assert row['start_seconds'] == pytest.approx(row['start_frame']/FS)
            assert row['end_seconds_exclusive'] == pytest.approx(cursor/FS)
            assert row['duration_seconds'] == pytest.approx(frames/FS)
            reconstructed.append(payload)
        assert cursor == frame_count
        assert b''.join(reconstructed) == payloads[name]
        assert read_pcm(data/name/'audio_000000.wav')[1] == payloads[name]
    assert not (output/'tails'/'test'/'test_recording'/'tail.wav').exists()

    for split, filename in [('train', 'train.csv'), ('val', 'val.csv'),
                            ('test', 'test.csv'), ('exclude', 'excluded.csv')]:
        rows = read_csv(output/filename)
        expected = [row for row in full if row['split'] == split]
        assert {row['sample_id'] for row in rows} == {row['sample_id'] for row in expected}
        assert all(row['is_full_segment'].lower() in {'true', '1'} for row in rows)
        assert all(not row['wav_file'].startswith('tails/') for row in rows)
    assert len(read_csv(output/'segments.csv')) == len(full)
    assert len(read_csv(output/'tails.csv')) == len(tails)
    assert summary['source_frame_count'] == sum(frames for _, _, frames in specs)
    assert summary['full_duration_s'] == pytest.approx(25)
    assert summary['tail_duration_s'] == pytest.approx(128/FS)


def test_events_inherit_at_window_start_and_preserve_internal_boundary_times(tmp_path):
    data = tmp_path/'data'; data.mkdir()
    annotations = [event('start', 0, 'machining', 3),
                   event('inside', .015, 'tool_change', 3),
                   event('boundary', .020, 'machining', 3)]
    write_recording(data, 'clip', 4*441 + 97, source='original_clip', source_start=100,
                    events=annotations, playback_ms=0)
    write_recording(data, 'val', 441)
    write_recording(data, 'test', 441)
    rows = [assignment('clip', 'train', 'original_clip'), assignment('val', 'val'), assignment('test', 'test')]
    splits = tmp_path/'split.csv'; write_split_csv(splits, rows)
    output = tmp_path/'out'
    build_dataset_segments(data, splits, output, segment_seconds=.01)
    full = read_json(output/'segments.json')
    clip_rows = sorted([row for row in full if row['recording_id'] == 'clip'], key=lambda row: row['start_frame'])
    assert len(clip_rows) == 4
    assert clip_rows[0]['machine_state'] == 'machining'
    assert clip_rows[0]['machine_state_status'] == 'annotated_constant'
    mixed = clip_rows[1]
    assert mixed['machine_state'] is None and mixed['machine_state_status'] == 'mixed'
    pieces = mixed['event_intervals']
    assert [piece['machine_state'] for piece in pieces] == ['machining', 'tool_change']
    assert [piece['start_seconds'] for piece in pieces] == pytest.approx([0, .005])
    assert [piece['end_seconds_exclusive'] for piece in pieces] == pytest.approx([.005, .01])
    assert [piece['source_start_seconds'] for piece in pieces] == pytest.approx([.01, .015])
    assert [piece['original_recording_start_seconds'] for piece in pieces] == pytest.approx([100.01, 100.015])
    for row in clip_rows[2:]:
        assert row['machine_state'] == 'machining'
        assert row['machine_state_status'] == 'annotated_constant'
    assert all(row['quality_score'] == 3 and row['quality_status'] == 'annotated_constant' for row in clip_rows)
    tail = next(row for row in read_json(output/'tails.json') if row['recording_id'] == 'clip')
    assert tail['machine_state'] == 'machining' and tail['quality_score'] == 3
    csv_row = next(row for row in read_csv(output/'train.csv') if int(row['start_frame']) == 441)
    assert json.loads(csv_row['event_intervals']) == pieces

    # SourceRecording's historical 1400-ms offset must not shift local events
    # when the current clip's playback.json explicitly declares zero offset.
    assert clip_rows[0]['event_intervals'][0]['source_start_seconds'] == 0
    unannotated = [row for row in full if row['recording_id'] in {'val', 'test'}]
    assert all(row['machine_state'] is None and row['quality_score'] is None for row in unannotated)
    assert all(row['quality_status'] == 'unannotated' for row in unannotated)


def test_partial_annotation_keeps_unknown_time_and_does_not_fill_missing_values_with_zero(tmp_path):
    data = tmp_path/'data'; data.mkdir()
    write_recording(data, 'train', 882, events=[event('late', .005, 'machining', 3)], playback_ms=0)
    write_recording(data, 'val', 441, events=[event('missing', 0, None, None)], playback_ms=0)
    write_recording(data, 'test', 441)
    rows = [assignment(name, name) for name in ['train', 'val', 'test']]
    splits = tmp_path/'split.csv'; write_split_csv(splits, rows)
    output = tmp_path/'out'
    build_dataset_segments(data, splits, output, segment_seconds=.01)
    full = read_json(output/'segments.json')
    first = next(row for row in full if row['recording_id'] == 'train' and row['start_frame'] == 0)
    assert first['machine_state'] is None and first['quality_score'] is None
    assert first['machine_state_status'] == first['quality_status'] == 'partially_annotated'
    assert first['event_intervals'][0]['machine_state'] is None
    assert first['event_intervals'][0]['quality_score'] is None
    second = next(row for row in full if row['recording_id'] == 'train' and row['start_frame'] == 441)
    assert second['machine_state'] == 'machining' and second['quality_score'] == 3
    for row in full:
        if row['recording_id'] != 'train':
            assert row['machine_state'] is None and row['quality_score'] is None
            assert row['machine_state_status'] == row['quality_status'] == 'unannotated'
    for row in read_csv(output/'val.csv'):
        assert row['machine_state'] == '' and row['quality_score'] == ''


def test_same_source_across_active_splits_is_rejected(tmp_path):
    data = tmp_path/'data'; data.mkdir()
    write_recording(data, 'clip_train', 441, source='shared_original')
    write_recording(data, 'clip_val', 441, source='shared_original')
    write_recording(data, 'test', 441)
    rows = [assignment('clip_train', 'train', 'shared_original'),
            assignment('clip_val', 'val', 'shared_original'), assignment('test', 'test')]
    splits = tmp_path/'split.csv'; write_split_csv(splits, rows)
    with pytest.raises(ValueError, match='shared_original.*both train and val'):
        build_dataset_segments(data, splits, tmp_path/'out', segment_seconds=.01)


def test_forged_source_id_is_rejected_against_metadata(tmp_path):
    data = tmp_path/'data'; data.mkdir()
    write_recording(data, 'clip_train', 441, source='actual_source')
    write_recording(data, 'val', 441)
    write_recording(data, 'test', 441)
    rows = [assignment('clip_train', 'train', 'forged_source'),
            assignment('val', 'val'), assignment('test', 'test')]
    splits = tmp_path/'split.csv'; write_split_csv(splits, rows)
    with pytest.raises(ValueError) as raised:
        build_dataset_segments(data, splits, tmp_path/'out', segment_seconds=.01)
    assert 'source' in str(raised.value).lower()
    assert 'actual_source' in str(raised.value) and 'forged_source' in str(raised.value)


def test_distinct_clips_of_one_source_in_the_same_split_keep_distinct_outputs(tmp_path):
    data = tmp_path/'data'; data.mkdir()
    first = write_recording(data, 'clip_one', 441, source='same_original', source_start=0, seed=10)
    second = write_recording(data, 'clip_two', 441, source='same_original', source_start=10, seed=200)
    write_recording(data, 'val', 441)
    write_recording(data, 'test', 441)
    rows = [assignment('clip_one', 'train', 'same_original'), assignment('clip_two', 'train', 'same_original'),
            assignment('val', 'val'), assignment('test', 'test')]
    splits = tmp_path/'split.csv'; write_split_csv(splits, rows)
    output = tmp_path/'out'
    build_dataset_segments(data, splits, output, segment_seconds=.01)
    train_rows = [row for row in read_json(output/'segments.json') if row['split'] == 'train']
    assert len(train_rows) == 2
    assert len({row['sample_id'] for row in train_rows}) == 2
    assert len({row['wav_file'] for row in train_rows}) == 2
    expected = {'clip_one': first, 'clip_two': second}
    for row in train_rows:
        assert row['source_recording_id'] == 'same_original'
        assert read_pcm(output/row['wav_file'])[1] == expected[row['recording_id']]


def test_nonempty_output_is_rejected_without_modifying_existing_files(tmp_path):
    data, splits, _, _ = write_inputs(tmp_path, [('train', 'train', 441), ('val', 'val', 441), ('test', 'test', 441)])
    output = tmp_path/'out'; output.mkdir()
    marker = output/'keep.txt'; marker.write_bytes(b'existing dataset result')
    with pytest.raises((FileExistsError, ValueError)):
        build_dataset_segments(data, splits, output, segment_seconds=.01)
    assert marker.read_bytes() == b'existing dataset result'
    assert list(output.iterdir()) == [marker]
