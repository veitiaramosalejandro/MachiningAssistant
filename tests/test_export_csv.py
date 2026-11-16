"""Standalone index exports preserve manual labels and never rewrite source data."""
from __future__ import annotations

import csv
import json
import sys
import wave
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from Dataset.Preprocessing.export_csv import export_csv_indexes


def write_csv(path, rows, fields=None):
    fields = list(rows[0]) if fields is None else fields
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_current_splits(root, rows):
    for split in ('train', 'val', 'test', 'exclude'):
        name = 'excluded.csv' if split == 'exclude' else f'{split}.csv'
        write_csv(root/name, [r for r in rows if r['split'] == split], fields=list(rows[0]))


def fixture_dataset(root, names=('good', 'sufficient', 'bad', 'very_bad'), current=True):
    root.mkdir()
    (root/'labels.json').write_text(json.dumps({str(i): name for i, name in enumerate(names)}))
    rows = []
    for split, indices in [('train', range(len(names))), ('val', [0]), ('test', [len(names)-1]), ('exclude', [1])]:
        for index in indices:
            sample = f'{split}_{index}'
            relative = f'{split}/{sample}.wav'
            wav_path = root/relative; wav_path.parent.mkdir(exist_ok=True)
            with wave.open(str(wav_path), 'wb') as wav:
                wav.setnchannels(2); wav.setsampwidth(2); wav.setframerate(44100)
                wav.writeframes(bytes([index, 0, 7, 0]*3))
            rows.append({'sample_id': sample, 'base_sample_id': f'base_{sample}',
                         'recording_id': f'record_{sample}', 'source_recording_id': f'source_{split}',
                         'split': split, 'wav_file': relative, 'frame_count': 3,
                         'label': names[index], 'label_index': index,
                         'original_quality_score': 3,
                         'original_event_intervals': [{'start_seconds': 0., 'quality_score': 3}]})
    (root/'samples.json').write_text(json.dumps(rows))
    write_csv(root/'samples.csv', rows)
    if current:
        write_current_splits(root, rows)
    return rows


def test_current_renamed_and_merged_labels_win_over_generation_metadata(tmp_path):
    root, output = tmp_path/'data', tmp_path/'review'
    rows = fixture_dataset(root)
    (root/'labels.json').write_text(json.dumps({'0': 'original', '1': 'modified'}))
    for row in rows:
        row['label_index'] = int(row['label_index'] != 0)
        row['label'] = ('original', 'modified')[row['label_index']]
    write_current_splits(root, rows)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    summary = export_csv_indexes(root, output)
    exported = read_csv(output/'samples.csv')
    assert summary['label_authority'] == 'current_split_csvs'
    assert summary['generation_label_differences_preserved'] == len(rows)
    assert summary['sample_count'] == 7
    assert summary['split_counts'] == {'train': 4, 'val': 1, 'test': 1, 'exclude': 1}
    assert {r['label'] for r in exported} == {'original', 'modified'}
    assert {int(r['label_index']) for r in exported} == {0, 1}
    assert 'very_bad' in (root/'samples.json').read_text()
    assert 'very_bad' in (root/'samples.csv').read_text()
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert summary['relative_path_base'] == str(root.resolve())
    assert summary['export_directory_is_standalone_dataset'] is False
    for row in exported:
        assert (root/row['wav_file']).is_file()
        assert not (output/row['wav_file']).exists()
    assert not list(output.rglob('*.wav'))
    assert 'not a standalone dataset' in (output/'README.txt').read_text()


def test_all_active_csvs_absent_can_export_from_samples_json(tmp_path):
    root, output = tmp_path/'data', tmp_path/'review'
    original = fixture_dataset(root, names=('normal', 'other'), current=False)
    summary = export_csv_indexes(root, output)
    assert summary['label_authority'] == 'generation_metadata_no_current_split_csvs'
    assert summary['generation_label_differences_preserved'] == 0
    exported = read_csv(output/'samples.csv')
    assert len(exported) == len(original)
    assert json.loads(exported[0]['original_event_intervals']) == original[0]['original_event_intervals']
    assert not (root/'train.csv').exists()


def test_partial_current_csv_set_is_not_silently_rebuilt(tmp_path):
    root, output = tmp_path/'data', tmp_path/'review'
    fixture_dataset(root)
    (root/'val.csv').unlink()
    with pytest.raises(FileNotFoundError, match='Incomplete current CSV set'):
        export_csv_indexes(root, output)
    assert not output.exists()


def test_nonempty_output_is_preserved(tmp_path):
    root, output = tmp_path/'data', tmp_path/'review'
    fixture_dataset(root); output.mkdir()
    marker = output/'keep.txt'; marker.write_text('existing review')
    with pytest.raises(FileExistsError, match='not empty'):
        export_csv_indexes(root, output)
    assert marker.read_text() == 'existing review'
    assert not (output/'train.csv').exists()


@pytest.mark.parametrize('problem', ['source', 'base'])
def test_cross_split_source_or_base_is_rejected(tmp_path, problem):
    root, output = tmp_path/'data', tmp_path/'review'
    rows = fixture_dataset(root)
    heldout = next(row for row in rows if row['split'] == 'test')
    key = 'source_recording_id' if problem == 'source' else 'base_sample_id'
    heldout[key] = rows[0][key]
    (root/'samples.json').write_text(json.dumps(rows))
    write_current_splits(root, rows)
    with pytest.raises(ValueError, match=f'A {key} crosses'):
        export_csv_indexes(root, output)


def test_current_label_must_match_current_class_mapping(tmp_path):
    root, output = tmp_path/'data', tmp_path/'review'
    rows = fixture_dataset(root)
    rows[0]['label'] = 'not_the_mapped_class'
    write_current_splits(root, rows)
    with pytest.raises(ValueError, match='current label/index does not match'):
        export_csv_indexes(root, output)


def test_current_split_cannot_be_changed_against_frozen_source_metadata(tmp_path):
    root, output = tmp_path/'data', tmp_path/'review'
    rows = fixture_dataset(root)
    rows[0]['split'] = 'val'
    write_current_splits(root, rows)
    with pytest.raises(ValueError, match='identity differs from samples.json'):
        export_csv_indexes(root, output)
