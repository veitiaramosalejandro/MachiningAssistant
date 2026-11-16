import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import experiment
import make_report
from probe_at16 import historical_pilot_ids
from Dataset.Preprocessing.dataset_splits import read_split_csv, write_split_csv


def assignments(train='dataset1', val='dataset2'):
    return [
        {'recording_id': train, 'source_recording_id': train, 'split': 'train'},
        {'recording_id': val, 'source_recording_id': val, 'split': 'val'},
        {'recording_id': 'dataset3', 'source_recording_id': 'dataset3', 'split': 'test'},
        {'recording_id': 'dataset4', 'source_recording_id': 'dataset4', 'split': 'exclude'},
    ]


def metadata():
    return [{'recording_id': f'dataset{record}', 'interval_id': f'dataset{record}:{label}'}
            for record in range(1, 6) for label in range(3)]


def test_csv_is_reread_and_excluded_or_unlisted_windows_do_not_enter_training(tmp_path):
    path = tmp_path/'split.csv'
    write_split_csv(path, assignments())
    _, first, first_indices, provenance = experiment.resolve_training_split(metadata(), path)
    assert first['train'] == ['dataset1']
    assert provenance['excluded_recordings'] == ['dataset4']
    assert provenance['unlisted_recordings'] == ['dataset5']
    assert provenance['ignored_recordings'] == ['dataset4', 'dataset5']
    assert provenance['ignored_window_count'] == 6
    np.testing.assert_array_equal(first_indices['train'], [0, 1, 2])
    write_split_csv(path, assignments(train='dataset2', val='dataset1'))
    _, second, second_indices, _ = experiment.resolve_training_split(metadata(), path)
    assert second['train'] == ['dataset2']
    np.testing.assert_array_equal(second_indices['train'], [3, 4, 5])


@pytest.mark.parametrize('change,match', [
    ('unknown', 'absent from prepared metadata'),
    ('source', 'source_recording_id must equal recording_id'),
])
def test_unknown_or_invented_msm_recording_identity_is_rejected(tmp_path, change, match):
    rows = assignments()
    if change == 'unknown':
        rows[-1] = {'recording_id': 'missing', 'source_recording_id': 'missing', 'split': 'exclude'}
    else:
        rows[0]['source_recording_id'] = 'invented_group'
    path = tmp_path/'split.csv'
    write_split_csv(path, rows)
    with pytest.raises(ValueError, match=match):
        experiment.resolve_training_split(metadata(), path)


@dataclass
class TinyConfig:
    sequence_length: int = 4
    input_channels: int = 2
    latent_dim: int = 16


class TinyClassifier(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head = torch.nn.Linear(config.sequence_length * config.input_channels, 3)

    def forward(self, x, reconstruct=False):
        return {'logits': self.head(x.flatten(1))}


def tiny_loss(output, labels, x, *, class_weights, **kwargs):
    loss = torch.nn.functional.cross_entropy(output['logits'], labels, weight=class_weights)
    return {'loss': loss, 'ce_loss': loss}


def test_small_training_uses_csv_snapshot_and_report_ignores_stale_cache_splits(tmp_path, monkeypatch):
    prepared = tmp_path/'prepared'
    prepared.mkdir()
    x = np.arange(15*4*2, dtype=np.float32).reshape(15, 4, 2)
    x[9:] = 1000000  # excluded and unlisted data must not affect normalization
    np.save(prepared/'windows.npy', x)
    np.save(prepared/'labels.npy', np.tile(np.arange(3), 5))
    (prepared/'windows.json').write_text(json.dumps(metadata()))
    (prepared/'splits.json').write_text('this stale file must never be read')
    preprocessing = {'target_sample_rate_hz': 48000, 'sequence_length': 4,
                     'windows_per_interval_limit': 1, 'label_edge_trim_seconds_each_side': .02}
    (prepared/'preprocessing.json').write_text(json.dumps(preprocessing))
    (prepared/'summary.json').write_text(json.dumps({
        'recording_count': 5, 'labelled_intervals': 15, 'retained_intervals': 15,
        'windows': 15, 'excluded_count': 0, 'actual_audio_duration_seconds': 1,
        'split_recordings': {'train': 999, 'val': 888, 'test': 777},
        'split_window_class_counts': {'train': [999]*3, 'val': [888]*3, 'test': [777]*3},
    }))
    split_csv = tmp_path/'split.csv'
    outputs = []
    for iteration, rows in enumerate([assignments(), assignments(train='dataset2', val='dataset1')]):
        write_split_csv(split_csv, rows)
        output = tmp_path/f'run{iteration}'
        monkeypatch.setattr(sys, 'argv', ['train', '--prepared', str(prepared), '--split-csv', str(split_csv),
                                         '--output', str(output), '--epochs', '1', '--threads', '1'])
        experiment.run_training(TinyClassifier, TinyConfig, tiny_loss, 'synthetic_test')
        outputs.append(output)
        saved = json.loads((output/'data_split.json').read_text())
        assert saved['train'] == [rows[0]['recording_id']]
        assert read_split_csv(output/'dataset_splits_used.csv') == rows
        report = json.loads((output/'classification_report.json').read_text())
        assert report['split_window_counts'] == {'train': 3, 'val': 3, 'test': 3}
        assert report['ignored_recordings'] == ['dataset4', 'dataset5']
        assert report['split_csv_source'] == str(split_csv.resolve())
        state = torch.load(output/'best_classifier.pt', map_location='cpu', weights_only=True)
        expected_indices = np.arange(3) if iteration == 0 else np.arange(3, 6)
        expected_mean, expected_scale = experiment.train_normalizer(x, expected_indices)
        np.testing.assert_allclose(state['normalizer']['mean'], expected_mean)
        np.testing.assert_allclose(state['normalizer']['scale'], expected_scale)
    assert read_split_csv(outputs[0]/'dataset_splits_used.csv') == assignments()

    first = make_report.load_model('first', outputs[0])
    second = make_report.load_model('second', outputs[1])
    with pytest.raises(ValueError, match='different actual source-recording splits'):
        make_report.validate_comparable_models([first, second])

    comparison = tmp_path/'comparison.json'
    comparison.write_text(json.dumps({'comparison': [{'dimension': 'test', 'at': 'a', 'msm': 'b', 'implication': 'c'}]}))
    report_dir = tmp_path/'rendered'
    monkeypatch.setattr(make_report, 'draw_confusion_matrices', lambda *args: None)
    monkeypatch.setattr(make_report, 'draw_learning_curves', lambda *args: None)
    monkeypatch.setattr(sys, 'argv', ['report', '--cnn-dir', str(outputs[0]), '--lstm-dir', str(outputs[0]),
                                    '--prepared', str(prepared), '--comparison', str(comparison), '--output', str(report_dir)])
    make_report.main()
    content = (report_dir/'report.html').read_text()
    assert '本轮实际选择3次录制、9个窗口' in content
    assert 'dataset4, dataset5' in content


def test_historical_probe_reads_only_original_manifest(tmp_path):
    artifacts = tmp_path/'artifacts'
    artifacts.mkdir()
    ids = [f'original_{i:02d}' for i in range(16)]
    manifest = artifacts/'data_manifest.json'
    manifest.write_text(json.dumps([{'recording_id': value} for value in reversed(ids)]))
    (tmp_path/'dataset_splits.csv').write_text('unrelated editable training assignments')
    assert historical_pilot_ids(tmp_path) == ids
    manifest.write_text(json.dumps([{'recording_id': value} for value in ids[:-1]]))
    with pytest.raises(ValueError, match='16 distinct original recording IDs'):
        historical_pilot_ids(tmp_path)
