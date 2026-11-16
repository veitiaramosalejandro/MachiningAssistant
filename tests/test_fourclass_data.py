"""Complete-waveform preprocessing and editable-label/split integrity tests."""
from __future__ import annotations

import csv
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from Dataset.Preprocessing import fourclass_data as data


def write_wav(path, signal, rate=44100):
    path.parent.mkdir(parents=True, exist_ok=True)
    signal = np.asarray(signal, dtype='<i2')
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(signal.shape[1]); wav.setsampwidth(2); wav.setframerate(rate)
        wav.writeframes(signal.tobytes())


def write_tables(root, rows, frozen=True):
    for split in data.SPLITS:
        with (root/f'{split}.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader()
            writer.writerows(row for row in rows if row['split'] == split)
    if frozen:
        (root/'samples.json').write_text(json.dumps(rows), encoding='utf-8')


def write_combined(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def make_dataset(root, names=('class_a', 'class_b'), frames=8, frozen=True):
    root.mkdir()
    (root/'labels.json').write_text(json.dumps({str(i): name for i, name in enumerate(names)}))
    rows = []
    for split, indices in [('train', range(len(names))), ('val', [0]), ('test', [len(names)-1])]:
        for i in indices:
            identifier = f'{split}_{i}'
            pcm = np.column_stack((np.arange(frames)*2 + i*4, np.full(frames, 7))).astype(np.int16)
            path = f'{split}/{identifier}.wav'
            write_wav(root/path, pcm)
            rows.append({'sample_id': identifier, 'base_sample_id': 'base_'+identifier,
                         'recording_id': 'record_'+identifier, 'source_recording_id': 'source_'+split,
                         'split': split, 'label': names[i], 'label_index': i, 'wav_file': path,
                         'frame_count': frames, 'start_frame': 0, 'end_frame_exclusive': frames,
                         'start_seconds': 0., 'end_seconds_exclusive': frames/44100,
                         'duration_seconds': frames/44100, 'peak_amplitude_pcm': 1000*i,
                         'phase_radians': float(i)})
    write_tables(root, rows, frozen)
    return rows


@pytest.mark.parametrize('class_count', [2, 3, 5])
def test_dynamic_class_counts_and_real_single_variants_are_supported(tmp_path, class_count):
    root = tmp_path/'dataset'
    names = tuple(f'state_{i}' for i in range(class_count))
    expected = make_dataset(root, names, frozen=(class_count != 3))
    rows = data.load_manifest(root, valid_frames=8)
    assert data.load_class_names(root) == names
    assert len(rows) == len(expected) == class_count+2
    assert {row['label_index'] for row in rows if row['split'] == 'train'} == set(range(class_count))
    assert len([row for row in rows if row['split'] == 'val']) == 1
    assert 'peak_amplitude_pcm' not in rows[0] and 'phase_radians' not in rows[0]
    assert isinstance(rows[0]['label_index'], int) and isinstance(rows[0]['duration_seconds'], float)


def test_relabel_csv_and_labels_without_rewriting_generated_label_metadata(tmp_path):
    root = tmp_path/'dataset'; rows = make_dataset(root, data.CLASS_NAMES)
    assert len(data.load_class_names(root)) == 4
    data.load_manifest(root, valid_frames=8)
    (root/'labels.json').write_text(json.dumps({'0': 'original', '1': 'injected'}))
    for row in rows:
        row['label_index'] = int(row['label_index'] != 0)
        row['label'] = ('original', 'injected')[row['label_index']]
    write_tables(root, rows, frozen=False)
    loaded = data.load_manifest(root, valid_frames=8)
    assert data.load_class_names(root) == ('original', 'injected')
    assert {row['label'] for row in loaded} == {'original', 'injected'}
    assert 'very_bad' in (root/'samples.json').read_text()


@pytest.mark.parametrize('mapping', [
    {'0': 'only_one'}, {'0': 'a', '2': 'b'}, {'0': 'same', '1': 'same'}, {'0': '', '1': 'b'},
])
def test_invalid_label_maps_are_rejected(tmp_path, mapping):
    (tmp_path/'labels.json').write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match='labels.json'):
        data.load_class_names(tmp_path)


def test_training_must_cover_configured_classes(tmp_path):
    root = tmp_path/'dataset'; rows = make_dataset(root)
    rows = [r for r in rows if r['sample_id'] != 'train_1']
    write_tables(root, rows)
    with pytest.raises(ValueError, match='every configured class'):
        data.load_manifest(root, valid_frames=8)


def test_full_five_second_pcm_and_exact_44_frame_cnn_padding(tmp_path):
    n = data.VALID_FRAMES
    raw = np.column_stack((np.arange(n) % 201 - 100, np.arange(n) % 503 - 251)).astype(np.int16)
    path = tmp_path/'five_seconds.wav'; write_wav(path, raw)
    rate, pcm = data.read_wav(path, expected_frames=n)
    assert rate == 44100 and pcm.dtype == np.float32
    np.testing.assert_array_equal(pcm, raw)
    normalizer = {'mean_pcm': [10., -20.], 'scale_pcm': [2., 4.]}
    cnn = data.prepare_signal(pcm, normalizer, data.model_input_length('CNN1D-VAE'))
    lstm = data.prepare_signal(pcm, normalizer, data.model_input_length('LSTM-VAE'))
    expected = ((pcm.astype(np.float64)-[10., -20.])/[2., 4.]).astype(np.float32)
    assert cnn.shape == (220544, 2) and lstm.shape == (220500, 2)
    np.testing.assert_array_equal(cnn[:n], expected)
    np.testing.assert_array_equal(lstm, expected)
    np.testing.assert_array_equal(cnn[n:], np.zeros((44, 2), np.float32))
    with pytest.raises(ValueError, match='complete'):
        data.prepare_signal(pcm[:2048], normalizer, 220544)
    with pytest.raises(ValueError, match='exactly'):
        data.read_wav(path, expected_frames=2048)


def test_training_only_normalizer_uses_raw_counts_and_constant_channel_floor(tmp_path):
    root = tmp_path/'dataset'; make_dataset(root)
    rows = data.load_manifest(root, valid_frames=8)
    training = [row for row in rows if row['split'] == 'train']
    normalizer = data.fit_normalizer(root, training)
    expected = np.concatenate([data.read_wav(root/r['wav_file'])[1].astype(np.float64) for r in training])
    np.testing.assert_allclose(normalizer['mean_pcm'], expected.mean(axis=0))
    np.testing.assert_allclose(normalizer['scale_pcm'], np.maximum(expected.std(axis=0), 1.))
    assert normalizer['scale_pcm'][1] == 1.
    held_out = [row for row in rows if row['split'] != 'train']
    for row in held_out:
        write_wav(root/row['wav_file'], np.full((8, 2), 30000, np.int16))
    assert data.fit_normalizer(root, training) == normalizer
    with pytest.raises(ValueError, match='training rows only'):
        data.fit_normalizer(root, training+held_out)


def test_lazy_dataset_reads_only_selected_wav_and_returns_long_label(tmp_path, monkeypatch):
    root = tmp_path/'dataset'; make_dataset(root, frames=65)
    rows = data.load_manifest(root, valid_frames=65)
    normalizer = {'mean_pcm': [1., 2.], 'scale_pcm': [2., 3.]}
    calls = []
    original = data.read_wav

    def tracked(path, expected_frames=None):
        calls.append(Path(path))
        return original(path, expected_frames)

    monkeypatch.setattr(data, 'read_wav', tracked)
    dataset = data.RawFourClassDataset(root, rows, normalizer, 128, valid_frames=65)
    assert calls == []
    signal, label = dataset[1]
    assert calls == [(root/rows[1]['wav_file']).resolve()]
    assert signal.shape == (128, 2) and signal.dtype == torch.float32
    assert label.dtype == torch.long and label.item() == rows[1]['label_index']
    assert torch.count_nonzero(signal[65:]) == 0


@pytest.mark.parametrize('problem', ['source', 'base', 'path'])
def test_source_base_leakage_and_duplicate_actual_paths_are_rejected(tmp_path, problem):
    root = tmp_path/'dataset'; rows = make_dataset(root)
    if problem == 'source':
        rows[-1]['source_recording_id'] = rows[0]['source_recording_id']
        message = 'source_recording_id'
    elif problem == 'base':
        rows[-1]['base_sample_id'] = rows[0]['base_sample_id']
        message = 'base samples cross'
    else:
        rows[1]['wav_file'] = rows[0]['wav_file']
        message = 'physical WAV path'
    write_tables(root, rows)
    with pytest.raises(ValueError, match=message):
        data.load_manifest(root, valid_frames=8)


def test_csv_source_identity_must_match_optional_generated_metadata(tmp_path):
    root = tmp_path/'dataset'; rows = make_dataset(root)
    rows[0]['source_recording_id'] = 'edited_source'
    write_tables(root, rows, frozen=False)
    with pytest.raises(ValueError, match='differs from frozen samples.json'):
        data.load_manifest(root, valid_frames=8)


def test_explicit_combined_csv_controls_labels_and_order_without_changing_wav_root(tmp_path, monkeypatch):
    root = tmp_path/'dataset'; rows = make_dataset(root)
    external = tmp_path/'indexes'; external.mkdir()
    chosen = external/'chosen.csv'
    changed = [dict(row) for row in rows]
    validation = next(row for row in changed if row['split'] == 'val')
    validation['label'], validation['label_index'] = 'class_b', 1
    changed[0], changed[1] = changed[1], changed[0]
    write_combined(chosen, changed)
    monkeypatch.chdir(external)
    actual = data.load_manifest(root, valid_frames=8, csv_path='chosen.csv')
    assert [r['sample_id'] for r in actual if r['split'] == 'train'] == ['train_1', 'train_0']
    assert next(r for r in actual if r['split'] == 'val')['label_index'] == 1
    original = data.load_manifest(root, valid_frames=8)
    assert next(r for r in original if r['split'] == 'val')['label_index'] == 0
    assert all((root/r['wav_file']).is_file() for r in actual)
    assert not (external/'train').exists()
    validation['label'], validation['label_index'] = 'class_a', 0
    write_combined(chosen, changed)
    reloaded = data.load_manifest(root, valid_frames=8, csv_path=chosen)
    assert next(r for r in reloaded if r['split'] == 'val')['label_index'] == 0


def test_one_combined_csv_is_sufficient_without_three_split_files(tmp_path):
    root = tmp_path/'dataset'; rows = make_dataset(root)
    path = root/'samples.csv'; write_combined(path, rows)
    for split in data.SPLITS:
        (root/f'{split}.csv').unlink()
    actual = data.load_manifest(root, valid_frames=8, csv_path=path)
    assert len(actual) == len(rows)
    assert {r['split'] for r in actual} == set(data.SPLITS)


def test_train_only_csv_reports_missing_validation_and_test(tmp_path):
    root = tmp_path/'dataset'; make_dataset(root)
    with pytest.raises(ValueError, match='missing: val, test'):
        data.load_manifest(root, valid_frames=8, csv_path=root/'train.csv')


def test_combined_csv_exclude_rows_are_not_loaded(tmp_path):
    root = tmp_path/'dataset'; rows = make_dataset(root)
    excluded = dict(rows[-1])
    excluded.update(sample_id='ignored', base_sample_id='ignored_base', recording_id='ignored_recording',
                    source_recording_id='ignored_source', split='exclude', wav_file='missing/excluded.wav')
    path = root/'all.csv'; write_combined(path, rows+[excluded])
    actual = data.load_manifest(root, valid_frames=8, csv_path=path)
    assert len(actual) == len(rows)
    assert 'ignored' not in {row['sample_id'] for row in actual}


def test_combined_csv_rejects_unknown_split(tmp_path):
    root = tmp_path/'dataset'; rows = make_dataset(root)
    rows[-1]['split'] = 'testing_typo'
    path = root/'all.csv'; write_combined(path, rows)
    with pytest.raises(ValueError, match='Unknown split'):
        data.load_manifest(root, valid_frames=8, csv_path=path)


def test_combined_csv_cannot_bypass_frozen_split_identity(tmp_path):
    root = tmp_path/'dataset'; rows = make_dataset(root)
    rows[0]['split'], rows[2]['split'] = 'val', 'train'
    path = root/'all.csv'; write_combined(path, rows)
    with pytest.raises(ValueError, match='differs from frozen samples.json'):
        data.load_manifest(root, valid_frames=8, csv_path=path)


def test_no_csv_argument_keeps_default_active_dataset_and_explicit_root_behavior(tmp_path, monkeypatch):
    root = tmp_path/'dataset'; make_dataset(root)
    original_resolver = data.resolve_dataset_dir
    monkeypatch.setattr(data, 'resolve_dataset_dir', lambda value=None: root if value is None else original_resolver(value))
    assert data.load_manifest(valid_frames=8) == data.load_manifest(root, valid_frames=8)
