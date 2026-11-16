"""Verify deterministic four-class synthesis and preservation of base signals."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys
import wave

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from Dataset.Preprocessing.build_fourclass import build_fourclass_dataset
from Dataset.Preprocessing.build_segments import build_dataset_segments
from Dataset.Preprocessing.dataset_splits import write_split_csv


FS = 44100
FRAMES = 441


def json_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def csv_read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def pcm_write(path, values):
    values = np.asarray(values, dtype='<i2')
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(2); wav.setsampwidth(2); wav.setframerate(FS)
        wav.writeframes(values.tobytes())


def pcm_read(path):
    with wave.open(str(path), 'rb') as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) == (FS, 2, 2, 'NONE')
        count = wav.getnframes()
        payload = wav.readframes(count)
    return np.frombuffer(payload, dtype='<i2').reshape(count, 2).copy()


@pytest.fixture
def config():
    value = json_read(PROJECT/'Dataset/Preprocessing/fourclass_config.json')
    value['segment_seconds'] = .01
    return value


@pytest.fixture
def original_segments(tmp_path):
    raw = tmp_path/'raw'; raw.mkdir()
    assignments = []
    for name, split, amplitudes in [('training_recording', 'train', [100, 300, 1000]),
                                     ('validation_recording', 'val', [150]),
                                     ('test_recording', 'test', [250])]:
        directory = raw/name; directory.mkdir()
        blocks = []
        t = np.arange(FRAMES)/FS
        for amplitude in amplitudes:
            left = np.rint(60*np.sin(2*np.pi*230*t) + 17).astype(np.int16)
            right = np.rint(amplitude*np.sin(2*np.pi*700*t) + 37).astype(np.int16)
            blocks.append(np.column_stack([left, right]))
        pcm_write(directory/'audio_000000.wav', np.concatenate(blocks))
        json_write(directory/'metadata.json', {'DurationSeconds': len(amplitudes)*.01,
                                               'Context': {'Machine': 'fixture', 'Operation': 'machining'}})
        json_write(directory/'playback.json', {'AudioOffsetMs': 0})
        json_write(directory/'events.json', {'schema_version': 1, 'time_reference': 'playback_seconds',
            'annotations': [{'id': name, 'timestamp_seconds': 0, 'machine_state': 'machining', 'quality': 3,
                             'event_changes': {'machine_state': True, 'quality': True}}]})
        assignments.append({'recording_id': name, 'source_recording_id': name, 'split': split})
    split_file = tmp_path/'source_split.csv'; write_split_csv(split_file, assignments)
    source = tmp_path/'original_segments'
    build_dataset_segments(raw, split_file, source, segment_seconds=.01)
    return source


def rows_by_base(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['base_sample_id']].append(row)
    return grouped


def payloads_by_identity(output):
    return {(row['recording_id'], row['start_frame'], row['label']): pcm_read(output/row['wav_file']).tobytes()
            for row in json_read(output/'samples.json')}


def test_four_classes_match_the_logged_injection_and_preserve_original_channels(original_segments, config, tmp_path):
    source = original_segments
    bases = json_read(source/'segments.json')
    output = tmp_path/'fourclass'
    summary = build_fourclass_dataset(source, output, config)
    samples = json_read(output/'samples.json')
    assert len(samples) == 4*len(bases) == 20
    assert Counter(row['label'] for row in samples) == {label: len(bases) for label in config['labels']}
    assert {row['label']: row['label_index'] for row in samples} == dict(zip(config['labels'], range(4)))
    assert summary['base_sample_count'] == len(bases)
    assert summary['sample_count'] == len(samples)
    assert len({row['sample_id'] for row in samples}) == len(samples)
    training_rms = [np.sqrt(np.mean(pcm_read(source/row['wav_file'])[:, 1].astype(np.float64)**2))
                    for row in bases if row['split'] == 'train']
    expected_reference = float(np.median(training_rms))
    assert summary['reference_rms_pcm'] == pytest.approx(expected_reference)
    assert summary['reference_sample_count'] == 3
    base_lookup = {row['base_sample_id']: row for row in bases}
    grouped = rows_by_base(samples)
    random_parameters = []
    t = np.arange(FRAMES)/FS
    for base_id, variants in grouped.items():
        base = base_lookup[base_id]
        raw = pcm_read(source/base['wav_file'])
        assert {row['split'] for row in variants} == {base['split']}
        assert {row['source_recording_id'] for row in variants} == {base['source_recording_id']}
        phases = {row['phase_radians'] for row in variants}
        multipliers = {row['jitter_multiplier'] for row in variants}
        assert len(phases) == len(multipliers) == 1
        phase, multiplier = next(iter(phases)), next(iter(multipliers))
        assert .9 <= multiplier <= 1.1
        random_parameters.append((phase, multiplier))
        for row in variants:
            result = pcm_read(output/row['wav_file'])
            assert result.shape == (FRAMES, 2)
            np.testing.assert_array_equal(result[:, 0], raw[:, 0])
            assert row['original_quality_score'] == 3
            assert row['original_machine_state'] == 'machining'
            assert row['reference_rms_pcm'] == pytest.approx(expected_reference)
            assert row['generation_frequency_hz'] == config['sine_frequency_hz'] == 1000
            assert row['injection_channel'] == 1
            ratio = config['peak_amplitude_ratios'][row['label_index']]
            amplitude = ratio*expected_reference*multiplier
            assert row['peak_amplitude_ratio'] == ratio
            assert row['peak_amplitude_pcm'] == pytest.approx(amplitude)
            path = Path(row['wav_file'])
            assert not path.is_absolute()
            assert path.parts[:3] == (row['split'], row['label'], row['recording_id'])
            assert path.name == Path(base['wav_file']).name
            if row['label'] == 'good':
                np.testing.assert_array_equal(result, raw)
            else:
                expected = np.rint(raw[:, 1].astype(np.float64) + amplitude*np.sin(2*np.pi*1000*t + phase))
                np.testing.assert_array_equal(result[:, 1], expected.astype(np.int16))
    assert len(set(random_parameters)) == len(bases)
    for split in ['train', 'val', 'test']:
        rows = csv_read(output/f'{split}.csv')
        assert {row['sample_id'] for row in rows} == {row['sample_id'] for row in samples if row['split'] == split}
        assert {row['split'] for row in rows} == {split}
    groups = {split: {row['source_recording_id'] for row in samples if row['split'] == split}
              for split in ['train', 'val', 'test']}
    assert not groups['train'] & groups['val']
    assert not groups['train'] & groups['test']
    assert not groups['val'] & groups['test']


def test_validation_and_test_signals_do_not_change_training_reference_or_training_variants(original_segments, config, tmp_path):
    source = original_segments
    first = tmp_path/'first'; second = tmp_path/'second'
    before = build_fourclass_dataset(source, first, config)
    for row in json_read(source/'segments.json'):
        if row['split'] != 'train':
            values = pcm_read(source/row['wav_file']).astype(np.int32)
            values[:, 1] *= 9
            pcm_write(source/row['wav_file'], values)
    after = build_fourclass_dataset(source, second, config)
    assert before['reference_rms_pcm'] == after['reference_rms_pcm']
    first_payloads = payloads_by_identity(first)
    second_payloads = payloads_by_identity(second)
    training_rows = [row for row in json_read(first/'samples.json') if row['split'] == 'train']
    for row in training_rows:
        key = (row['recording_id'], row['start_frame'], row['label'])
        assert first_payloads[key] == second_payloads[key]


def test_reordering_base_rows_does_not_change_generated_samples(original_segments, config, tmp_path):
    source = original_segments
    first = tmp_path/'first'; second = tmp_path/'second'
    build_fourclass_dataset(source, first, config)
    rows = json_read(source/'segments.json')
    json_write(source/'segments.json', list(reversed(rows)))
    build_fourclass_dataset(source, second, config)
    assert payloads_by_identity(first) == payloads_by_identity(second)
    first_parameters = {(row['recording_id'], row['start_frame'], row['label']):
                        (row['phase_radians'], row['jitter_multiplier'])
                        for row in json_read(first/'samples.json')}
    second_parameters = {(row['recording_id'], row['start_frame'], row['label']):
                         (row['phase_radians'], row['jitter_multiplier'])
                         for row in json_read(second/'samples.json')}
    assert first_parameters == second_parameters


def test_duplicate_base_is_rejected(original_segments, config, tmp_path):
    rows = json_read(original_segments/'segments.json')
    json_write(original_segments/'segments.json', [*rows, dict(rows[0])])
    with pytest.raises(ValueError):
        build_fourclass_dataset(original_segments, tmp_path/'out', config)


@pytest.mark.parametrize('field,value', [('source_recording_id', 'forged_source'), ('split', 'val')])
def test_forged_source_or_split_is_rejected_against_split_snapshot(original_segments, config, tmp_path, field, value):
    rows = json_read(original_segments/'segments.json')
    row = next(item for item in rows if item['split'] == 'train')
    row[field] = value
    json_write(original_segments/'segments.json', rows)
    with pytest.raises(ValueError):
        build_fourclass_dataset(original_segments, tmp_path/'out', config)


def test_same_source_cannot_cross_sets_even_if_row_and_csv_are_changed_together(original_segments, config, tmp_path):
    split_rows = csv_read(original_segments/'dataset_splits_used.csv')
    train_source = next(row['source_recording_id'] for row in split_rows if row['split'] == 'train')
    for row in split_rows:
        if row['split'] == 'val':
            row['source_recording_id'] = train_source
    write_split_csv(original_segments/'dataset_splits_used.csv', split_rows)
    rows = json_read(original_segments/'segments.json')
    for row in rows:
        if row['split'] == 'val':
            row['source_recording_id'] = train_source
    json_write(original_segments/'segments.json', rows)
    with pytest.raises(ValueError):
        build_fourclass_dataset(original_segments, tmp_path/'out', config)


def test_pcm_overflow_is_reported_instead_of_clipped(original_segments, config, tmp_path):
    for row in json_read(original_segments/'segments.json'):
        if row['split'] == 'train':
            values = pcm_read(original_segments/row['wav_file'])
            values[:, 1] = 30000
            pcm_write(original_segments/row['wav_file'], values)
    with pytest.raises(ValueError):
        build_fourclass_dataset(original_segments, tmp_path/'out', config)


def test_nonempty_output_is_rejected_without_touching_existing_result(original_segments, config, tmp_path):
    output = tmp_path/'out'; output.mkdir()
    marker = output/'keep.txt'; marker.write_bytes(b'previous output')
    with pytest.raises((FileExistsError, ValueError)):
        build_fourclass_dataset(original_segments, output, config)
    assert marker.read_bytes() == b'previous output'
    assert list(output.iterdir()) == [marker]
