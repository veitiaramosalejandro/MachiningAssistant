"""Small raw-WAV integration tests; no production-dataset training is run."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
import wave

import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import fourclass_training as training
from Dataset.Preprocessing.fourclass_data import RawFourClassDataset, load_manifest


VALID = 65
FS = 44100


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def model_classes(family, monkeypatch):
    directory = PROJECT/family
    model_name = '_classification_test_' + family.replace('-', '_')
    specification = importlib.util.spec_from_file_location(model_name, directory/'model.py')
    model_module = importlib.util.module_from_spec(specification)
    monkeypatch.setitem(sys.modules, model_name, model_module)
    specification.loader.exec_module(model_module)
    monkeypatch.setitem(sys.modules, 'model', model_module)
    name = model_name + '_classifier'
    specification = importlib.util.spec_from_file_location(name, directory/'classifier.py')
    classifier_module = importlib.util.module_from_spec(specification)
    monkeypatch.setitem(sys.modules, name, classifier_module)
    specification.loader.exec_module(classifier_module)
    return classifier_module.StateClassifier, model_module.ModelConfig


def make_data(root, class_count=3):
    root.mkdir()
    names = [f'state_{index}' for index in range(class_count)]
    (root/'labels.json').write_text(json.dumps({str(index): name for index, name in enumerate(names)}))
    (root/'dataset_summary.json').write_text(json.dumps({'synthetic_only': True, 'label_semantics': 'Temporary engineering fixture labels'}))
    t = np.arange(VALID)/VALID
    for split, selected in [('train', range(class_count)), ('val', [0]), ('test', [0, class_count-1])]:
        rows = []
        for index in selected:
            relative = Path(split)/f'{split}_{index}.wav'
            (root/relative).parent.mkdir(exist_ok=True)
            signal = np.column_stack([np.rint(100*np.sin(2*np.pi*(index+1)*t)+index),
                                      np.rint(220*np.cos(2*np.pi*(index+2)*t)+40*index)]).astype('<i2')
            with wave.open(str(root/relative), 'wb') as wav:
                wav.setnchannels(2); wav.setsampwidth(2); wav.setframerate(FS); wav.writeframes(signal.tobytes())
            rows.append({'sample_id': f'{split}_{index}', 'base_sample_id': f'{split}_base_{index}',
                         'recording_id': f'{split}_recording', 'source_recording_id': f'{split}_source',
                         'split': split, 'label': names[index], 'label_index': index, 'wav_file': relative.as_posix(),
                         'frame_count': VALID, 'start_frame': 0, 'end_frame_exclusive': VALID,
                         'start_seconds': 0., 'end_seconds_exclusive': VALID/FS, 'duration_seconds': VALID/FS})
        write_csv(root/f'{split}.csv', rows)
    return names


def call_train(classes, family, data, output, epochs, *extra):
    classifier, config = classes
    return training.run_training(classifier, config, family,
        ['--data-dir', str(data), '--output-dir', str(output), '--epochs', str(epochs),
         '--batch-size', '2', '--threads', '1', *extra], valid_frames=VALID)


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_reconstruction_loss_excludes_alignment_padding_and_keeps_valid_gradients():
    target = torch.zeros(2, 128, 2)
    prediction = torch.ones_like(target)
    prediction[:, VALID:] = 100
    prediction.requires_grad_()
    output = {'logits': torch.zeros(2, 2, requires_grad=True), 'mu': torch.ones(2, 2),
              'logvar': torch.zeros(2, 2), 'reconstruction': prediction}
    loss = training.joint_loss(output, torch.tensor([0, 1]), target, VALID)
    torch.testing.assert_close(loss['reconstruction_loss'], torch.tensor(1.))
    loss['loss'].backward()
    assert prediction.grad[:, :VALID].abs().sum() > 0
    assert prediction.grad[:, VALID:].count_nonzero() == 0


def test_metrics_keep_all_configured_classes_but_balance_only_present_true_classes():
    metrics = training.classification_metrics([0, 2], [0, 2], ['a', 'b', 'c'])
    assert metrics['accuracy'] == metrics['balanced_accuracy'] == 1
    assert metrics['macro_precision'] == pytest.approx(2/3)
    assert metrics['macro_recall'] == pytest.approx(2/3)
    assert metrics['macro_f1'] == pytest.approx(2/3)
    assert np.asarray(metrics['confusion_matrix']).shape == (3, 3)
    assert metrics['per_class'][1]['support'] == 0
    assert metrics['per_class'][1]['precision'] == metrics['per_class'][1]['recall'] == metrics['per_class'][1]['f1'] == 0


@pytest.mark.parametrize('class_count,truth,predicted', [
    (2, [0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1, 0, 1]),
    (3, [0, 0, 0, 0, 0, 0, 1, 1, 2], [0, 0, 0, 1, 1, 2, 1, 2, 0]),
    (5, [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3],
        [0, 0, 0, 0, 1, 2, 4, 1, 1, 0, 2, 3, 0]),
])
def test_acc_macro_precision_and_recall_match_sklearn_on_imbalanced_errors(class_count, truth, predicted):
    names = [f'class_{index}' for index in range(class_count)]
    metrics = training.classification_metrics(truth, predicted, names)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, labels=list(range(class_count)), average='macro', zero_division=0,
    )
    assert metrics['accuracy'] == pytest.approx(accuracy_score(truth, predicted))
    assert metrics['macro_precision'] == pytest.approx(precision)
    assert metrics['macro_recall'] == pytest.approx(recall)
    assert metrics['macro_f1'] == pytest.approx(f1)
    assert metrics['macro_precision'] != pytest.approx(metrics['accuracy'])
    assert metrics['macro_recall'] != pytest.approx(metrics['accuracy'])
    assert np.asarray(metrics['confusion_matrix']).shape == (class_count, class_count)
    formatted = training.format_classification_metrics(metrics)
    assert f'ACC={metrics["accuracy"]:.6f}' in formatted
    assert f'P(macro)={precision:.6f}' in formatted
    assert f'R(macro)={recall:.6f}' in formatted
    assert f'F1(macro)={f1:.6f}' in formatted


def reference_pr_from_matrix(matrix):
    truth, predicted = [], []
    for true_index, row in enumerate(matrix):
        for predicted_index, count in enumerate(row):
            truth.extend([true_index]*count)
            predicted.extend([predicted_index]*count)
    precision, recall, _, _ = precision_recall_fscore_support(
        truth, predicted, labels=list(range(len(matrix))), average='macro', zero_division=0,
    )
    return precision, recall


def test_complete_metrics_adds_missing_pr_from_saved_matrix_in_place():
    old = {'accuracy': 5/8, 'confusion_matrix': [[4, 2, 0], [1, 1, 0], [0, 0, 0]]}
    expected_p, expected_r = reference_pr_from_matrix(old['confusion_matrix'])
    training.complete_metrics(old)
    assert old['macro_precision'] == pytest.approx(expected_p)
    assert old['macro_recall'] == pytest.approx(expected_r)
    assert old['accuracy'] == 5/8
    assert training.complete_metrics(None) is None


def test_finite_large_gradients_are_clipped_when_float32_norm_overflows():
    parameter = torch.nn.Parameter(torch.zeros(4))
    parameter.grad = torch.full_like(parameter, 1e20)
    assert torch.isfinite(parameter.grad).all()
    assert not torch.isfinite(torch.linalg.vector_norm(parameter.grad))
    norm = training.clip_gradients([parameter], 5.)
    assert torch.isfinite(norm) and float(norm) == pytest.approx(2e20)
    torch.testing.assert_close(torch.linalg.vector_norm(parameter.grad), torch.tensor(5.))


@pytest.mark.parametrize('bad_value', [float('inf'), float('nan')])
def test_actual_nonfinite_gradients_still_stop_training(bad_value):
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([1., bad_value])
    with pytest.raises(RuntimeError, match='non-finite'):
        training.clip_gradients([parameter], 5.)


@pytest.mark.parametrize('family', ['CNN1D-VAE', 'LSTM-VAE'])
def test_real_model_training_saves_dynamic_class_checkpoint_and_best_predictions(tmp_path, monkeypatch, family):
    data = tmp_path/'data'; names = make_data(data, class_count=3)
    classes = model_classes(family, monkeypatch)
    output = tmp_path/'run'
    assert call_train(classes, family, data, output, 2, '--eval-every', '20') == 0
    last = load_checkpoint(output/'last.pt')
    assert last['task'] == 'classification' and last['model_family'] == family
    assert last['class_names'] == names and last['num_classes'] == 3
    assert last['sample_rate'] == FS and last['valid_frames'] == VALID
    assert last['model_config']['sequence_length'] == (128 if family == 'CNN1D-VAE' else VALID)
    if family == 'LSTM-VAE':
        assert last['model_config']['frame_size'] == 441
        assert last['lstm_frame_size'] == 441
        assert last['recurrent_steps'] == 1
        assert last['lstm_frame_padding_frames'] == 441-VALID
    assert last['training_state']['completed_epoch'] == 2
    assert last['training_state']['optimizer_state_dict']['state']
    history = read_json(output/'history.json')
    assert history[0]['validation_evaluated'] is False and history[0]['test_evaluated'] is False
    assert history[1]['validation_evaluated'] is True and history[1]['test_evaluated'] is True
    assert (output/'checkpoints/epoch_0002.pt').is_file()
    assert (output/'best.pt').is_file() and (output/'final.pt').is_file()
    metrics = read_json(output/'metrics.json')
    assert metrics['evaluation_weights'] == 'validation-selected best'
    assert Path(metrics['evaluation_checkpoint']) == output/'best.pt'
    assert np.asarray(metrics['test']['confusion_matrix']).shape == (3, 3)
    assert metrics['test']['sample_count'] == 2
    assert metrics['label_counts']['train'] == [1, 1, 1]
    with (output/'test_predictions.csv').open(encoding='utf-8-sig', newline='') as handle:
        predictions = list(csv.DictReader(handle))
    assert len(predictions) == 2
    assert {key for key in predictions[0] if key.startswith('probability_class_')} == {
        'probability_class_0', 'probability_class_1', 'probability_class_2'}
    best = load_checkpoint(output/'best.pt')
    classifier, config = classes
    restored = classifier(config(**best['model_config']), num_classes=3).eval()
    restored.load_state_dict(best['model_state_dict'])
    rows = [row for row in load_manifest(data, valid_frames=VALID) if row['split'] == 'test']
    dataset = RawFourClassDataset(data, rows, best['normalizer'], best['model_config']['sequence_length'], valid_frames=VALID)
    with torch.inference_mode():
        expected = restored(dataset[0][0][None], reconstruct=False)['logits'].softmax(-1)[0].numpy()
    np.testing.assert_allclose([float(predictions[0][f'probability_class_{index}']) for index in range(3)], expected, atol=1e-7, rtol=1e-6)
    assert metrics['evaluation_epoch'] == best['checkpoint_epoch']
    assert Path(predictions[0]['absolute_wav_file']).is_file()


@pytest.mark.parametrize('family', ['CNN1D-VAE', 'LSTM-VAE'])
def test_optimizer_and_rng_resume_match_uninterrupted_real_models(tmp_path, monkeypatch, family):
    data = tmp_path/'data'; make_data(data, class_count=4)
    classes = model_classes(family, monkeypatch)
    direct, first, resumed = tmp_path/'direct', tmp_path/'first', tmp_path/'resumed'
    assert call_train(classes, family, data, direct, 3, '--eval-every', '1') == 0
    assert call_train(classes, family, data, first, 1, '--eval-every', '1') == 0
    assert call_train(classes, family, data, resumed, 3, '--resume', str(first/'last.pt')) == 0
    reference = load_checkpoint(direct/'last.pt')
    restored = load_checkpoint(resumed/'last.pt')
    assert_nested_equal(reference['model_state_dict'], restored['model_state_dict'])
    assert_nested_equal(reference['training_state']['optimizer_state_dict'], restored['training_state']['optimizer_state_dict'])
    assert reference['normalizer'] == restored['normalizer']
    assert reference['training_state']['history'] == restored['training_state']['history']
    assert restored['training_state']['completed_epoch'] == 3
    assert restored['num_classes'] == 4
    assert (resumed/'best.pt').is_file() and (first/'last.pt').is_file()


@pytest.mark.parametrize('family', ['CNN1D-VAE', 'LSTM-VAE'])
def test_single_combined_csv_trains_and_resumes_without_three_split_tables(tmp_path, monkeypatch, family):
    data = tmp_path/'chosen dataset'; names = make_data(data, class_count=3)
    combined_rows = []
    for split in ('train', 'val', 'test'):
        path = data/f'{split}.csv'
        with path.open(encoding='utf-8-sig', newline='') as handle:
            combined_rows.extend(csv.DictReader(handle))
        path.unlink()
    csv_path = data/'samples.csv'
    write_csv(csv_path, combined_rows)
    classifier, config = model_classes(family, monkeypatch)
    first, resumed = tmp_path/'first', tmp_path/'resumed'
    assert training.run_training(classifier, config, family,
        ['--csv', str(csv_path), '--output-dir', str(first), '--epochs', '1',
         '--batch-size', '2', '--threads', '1'], valid_frames=VALID) == 0
    saved = load_checkpoint(first/'last.pt')
    settings = saved['training_state']['settings']
    assert settings['csv'] == str(csv_path.resolve())
    assert settings['data_dir'] == str(data.resolve())
    assert saved['class_names'] == names
    assert {row['sample_id'] for row in saved['training_state']['data_identity']['ordered_samples']} == {
        row['sample_id'] for row in combined_rows}
    assert read_json(first/'metrics.json')['test']['sample_count'] == 2
    assert not any((data/f'{split}.csv').exists() for split in ('train', 'val', 'test'))

    # A new invocation restores both paths without --csv or --data-dir.
    assert training.run_training(classifier, config, family,
        ['--resume', str(first/'last.pt'), '--output-dir', str(resumed), '--epochs', '2'],
        valid_frames=VALID) == 0
    continued = load_checkpoint(resumed/'last.pt')
    assert continued['training_state']['settings']['csv'] == str(csv_path.resolve())
    assert continued['training_state']['settings']['data_dir'] == str(data.resolve())
    assert continued['training_state']['completed_epoch'] == 2

    # Keeping the filename does not allow changing saved sample targets on resume.
    combined_rows[-1]['label_index'] = 1
    combined_rows[-1]['label'] = names[1]
    write_csv(csv_path, combined_rows)
    with pytest.raises(ValueError, match='identity|identities|changed|differs'):
        training.run_training(classifier, config, family,
            ['--resume', str(resumed/'last.pt'), '--output-dir', str(tmp_path/'changed'), '--epochs', '3'],
            valid_frames=VALID)


def test_legacy_checkpoint_without_csv_setting_resumes_with_original_split_files(tmp_path, monkeypatch):
    data = tmp_path/'legacy dataset'; make_data(data, class_count=3)
    classes = model_classes('CNN1D-VAE', monkeypatch)
    first, resumed = tmp_path/'first', tmp_path/'resumed'
    assert call_train(classes, 'CNN1D-VAE', data, first, 1) == 0
    legacy = load_checkpoint(first/'last.pt')
    legacy['training_state']['settings'].pop('csv', None)
    legacy_path = first/'legacy_without_csv.pt'
    torch.save(legacy, legacy_path)
    classifier, config = classes
    assert training.run_training(classifier, config, 'CNN1D-VAE',
        ['--resume', str(legacy_path), '--output-dir', str(resumed), '--epochs', '2'],
        valid_frames=VALID) == 0
    restored = load_checkpoint(resumed/'last.pt')
    assert restored['training_state']['settings']['csv'] is None
    assert restored['training_state']['settings']['data_dir'] == str(data.resolve())
    assert restored['training_state']['completed_epoch'] == 2


def test_resume_backfills_older_history_precision_recall_and_writes_mixed_csv(tmp_path, monkeypatch, capsys):
    data = tmp_path/'data'; make_data(data, class_count=3)
    classes = model_classes('CNN1D-VAE', monkeypatch)
    first, resumed = tmp_path/'first', tmp_path/'resumed'
    assert call_train(classes, 'CNN1D-VAE', data, first, 2, '--eval-every', '2') == 0
    legacy = load_checkpoint(first/'last.pt')
    state = legacy['training_state']
    for row in state['history']:
        for split in ('val', 'test'):
            row.pop(f'{split}_macro_precision', None)
            row.pop(f'{split}_macro_recall', None)
    expected = {}
    for entry in state['evaluations']:
        for key in ('validation', 'test'):
            metrics = entry[key]
            if metrics is not None:
                expected[(entry['epoch'], key)] = reference_pr_from_matrix(metrics['confusion_matrix'])
                metrics.pop('macro_precision', None)
                metrics.pop('macro_recall', None)
    legacy_path = first/'last_without_pr.pt'
    torch.save(legacy, legacy_path)
    assert call_train(classes, 'CNN1D-VAE', data, resumed, 3, '--resume', str(legacy_path)) == 0
    history = read_json(resumed/'history.json')
    assert [row['epoch'] for row in history] == [1, 2, 3]
    for split in ('val', 'test'):
        assert history[0][f'{split}_macro_precision'] is None
        assert history[0][f'{split}_macro_recall'] is None
        key = 'validation' if split == 'val' else 'test'
        precision, recall = expected[(2, key)]
        assert history[1][f'{split}_macro_precision'] == pytest.approx(precision)
        assert history[1][f'{split}_macro_recall'] == pytest.approx(recall)
        assert np.isfinite(history[2][f'{split}_macro_precision'])
        assert np.isfinite(history[2][f'{split}_macro_recall'])
    evaluations = read_json(resumed/'evaluations.json')
    for entry in evaluations:
        for key in ('validation', 'test'):
            metrics = entry[key]
            if metrics is not None:
                precision, recall = reference_pr_from_matrix(metrics['confusion_matrix'])
                assert metrics['macro_precision'] == pytest.approx(precision)
                assert metrics['macro_recall'] == pytest.approx(recall)
    with (resumed/'training_history.csv').open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        csv_rows = list(reader)
        assert {'val_macro_precision', 'val_macro_recall', 'test_macro_precision', 'test_macro_recall'} <= set(reader.fieldnames)
    assert len(csv_rows) == 3 and csv_rows[0]['val_macro_precision'] == ''
    assert float(csv_rows[1]['test_macro_recall']) == pytest.approx(expected[(2, 'test')][1])
    restored = load_checkpoint(resumed/'last.pt')
    assert restored['training_state']['completed_epoch'] == 3
    assert restored['training_state']['history'] == history
    console = capsys.readouterr().out
    assert all(label in console for label in ('ACC=', 'P(macro)=', 'R(macro)=', 'F1(macro)='))


def test_resume_rejects_legacy_weights_changed_mapping_and_changed_hyperparameters(tmp_path, monkeypatch):
    data = tmp_path/'data'; names = make_data(data, class_count=3)
    classes = model_classes('LSTM-VAE', monkeypatch)
    legacy = tmp_path/'legacy.pt'
    torch.save({'model_family': 'LSTM-VAE', 'task': 'msm'}, legacy)
    with pytest.raises(ValueError, match='matching classification checkpoint'):
        call_train(classes, 'LSTM-VAE', data, tmp_path/'legacy_out', 2, '--resume', str(legacy))
    first = tmp_path/'first'
    assert call_train(classes, 'LSTM-VAE', data, first, 1, '--eval-every', '1') == 0
    with pytest.raises(ValueError, match='setting mismatch for learning_rate'):
        call_train(classes, 'LSTM-VAE', data, tmp_path/'changed_lr', 2, '--resume', str(first/'last.pt'), '--lr', '.1')
    swapped = [names[1], names[0], names[2]]
    (data/'labels.json').write_text(json.dumps({str(index): name for index, name in enumerate(swapped)}))
    for split in ['train', 'val', 'test']:
        with (data/f'{split}.csv').open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row['label'] = swapped[int(row['label_index'])]
        write_csv(data/f'{split}.csv', rows)
    with pytest.raises(ValueError, match='class names, order or count'):
        call_train(classes, 'LSTM-VAE', data, tmp_path/'changed_mapping', 2, '--resume', str(first/'last.pt'))


def test_lstm_custom_frame_configuration_is_saved_and_resumed(tmp_path, monkeypatch):
    data = tmp_path/'data'; make_data(data, class_count=3)
    classes = model_classes('LSTM-VAE', monkeypatch)
    first, resumed = tmp_path/'first', tmp_path/'resumed'
    assert call_train(classes, 'LSTM-VAE', data, first, 1,
                      '--lstm-frame-size', '8', '--eval-every', '1') == 0
    saved = load_checkpoint(first/'last.pt')
    assert saved['model_config']['frame_size'] == saved['lstm_frame_size'] == 8
    assert saved['recurrent_steps'] == 9 and saved['lstm_frame_padding_frames'] == 7
    assert saved['model_state_dict']['vae.encoder.weight_ih_l0'].shape[1] == 16
    assert saved['model_state_dict']['vae.to_output.weight'].shape[0] == 16
    assert call_train(classes, 'LSTM-VAE', data, resumed, 2, '--resume', str(first/'last.pt')) == 0
    continued = load_checkpoint(resumed/'last.pt')
    assert continued['model_config']['frame_size'] == 8
    assert continued['training_state']['completed_epoch'] == 2
    with pytest.raises(ValueError, match='frame_size differs'):
        call_train(classes, 'LSTM-VAE', data, tmp_path/'wrong_frame', 3,
                   '--resume', str(resumed/'last.pt'), '--lstm-frame-size', '441')


def test_legacy_lstm_checkpoint_without_frame_field_keeps_frame_one(tmp_path, monkeypatch):
    data = tmp_path/'data'; make_data(data, class_count=3)
    classes = model_classes('LSTM-VAE', monkeypatch)
    first = tmp_path/'fresh_frame_one'
    assert call_train(classes, 'LSTM-VAE', data, first, 1,
                      '--lstm-frame-size', '1', '--eval-every', '1') == 0
    modern = load_checkpoint(first/'last.pt')
    assert modern['model_config']['frame_size'] == 1
    assert modern['model_state_dict']['vae.encoder.weight_ih_l0'].shape[1] == 2
    legacy = load_checkpoint(first/'last.pt')
    legacy['model_config'].pop('frame_size')
    for key in ('lstm_frame_size', 'recurrent_steps', 'lstm_frame_padding_frames'):
        legacy.pop(key, None)
    legacy['training_state']['settings'].pop('lstm_frame_size', None)
    legacy_path = first/'legacy_without_frame.pt'; torch.save(legacy, legacy_path)
    resumed = tmp_path/'resumed_legacy'
    assert call_train(classes, 'LSTM-VAE', data, resumed, 2, '--resume', str(legacy_path)) == 0
    restored = load_checkpoint(resumed/'last.pt')
    assert restored['model_config']['frame_size'] == restored['lstm_frame_size'] == 1
    assert restored['recurrent_steps'] == VALID and restored['lstm_frame_padding_frames'] == 0
    assert restored['model_state_dict']['vae.encoder.weight_ih_l0'].shape[1] == 2
    assert restored['training_state']['completed_epoch'] == 2
    with pytest.raises(ValueError, match='frame_size differs'):
        call_train(classes, 'LSTM-VAE', data, tmp_path/'invalid_conversion', 2,
                   '--resume', str(legacy_path), '--lstm-frame-size', '441')


def test_interruption_keeps_last_completed_epoch_and_can_resume(tmp_path, monkeypatch):
    data = tmp_path/'data'; make_data(data, class_count=4)
    classes = model_classes('CNN1D-VAE', monkeypatch)
    interrupted = tmp_path/'interrupted'
    original_step = torch.optim.AdamW.step
    calls = 0

    def interrupt_during_second_epoch(optimizer, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, 'step', interrupt_during_second_epoch)
    assert call_train(classes, 'CNN1D-VAE', data, interrupted, 3, '--eval-every', '1') == 130
    assert load_checkpoint(interrupted/'last.pt')['training_state']['completed_epoch'] == 1
    assert read_json(interrupted/'interruption.json')['last_completed_epoch'] == 1
    monkeypatch.setattr(torch.optim.AdamW, 'step', original_step)
    resumed = tmp_path/'resumed'
    assert call_train(classes, 'CNN1D-VAE', data, resumed, 3, '--resume', str(interrupted/'last.pt')) == 0
    assert load_checkpoint(resumed/'last.pt')['training_state']['completed_epoch'] == 3
