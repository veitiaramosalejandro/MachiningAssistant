"""Test the unified launcher without starting any training subprocess."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def app(monkeypatch):
    calls = []
    returncodes = [0, 0]
    timestamp_inputs = []
    training_roots = []

    def fake_run(command, **kwargs):
        index = len(calls)
        calls.append((list(command), kwargs))
        return subprocess.CompletedProcess(command, returncodes[index])

    # Patch before import as well, so no subprocess can run during these tests.
    monkeypatch.setattr(subprocess, 'run', fake_run)
    spec = importlib.util.spec_from_file_location('_model_selection_entrypoint', PROJECT/'main.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    def fake_training_root(output_root):
        root = Path(output_root).expanduser().resolve()
        timestamp_inputs.append(root)
        stamp = f'2026-11-10_20-00-00_{len(timestamp_inputs):06d}'
        result = root/'train'/stamp
        training_roots.append(result)
        return result

    monkeypatch.setattr(module, 'timestamped_training_root', fake_training_root)
    return SimpleNamespace(module=module, calls=calls, returncodes=returncodes,
                           timestamp_inputs=timestamp_inputs, training_roots=training_roots)


def option(command, name):
    return command[command.index(name) + 1]


@pytest.mark.parametrize('mode,input_mode', [('train', None), ('infer', 'file'), ('infer', 'stream')])
def test_selected_device_reaches_training_and_inference_process(app, prediction_files, tmp_path, mode, input_mode):
    checkpoint, wav = prediction_files
    argv = ['--mode', mode, '--model', 'cnn', '--device', 'cuda', '--output-dir', str(tmp_path/'runs')]
    if mode == 'infer':
        argv.extend(['--input-mode', input_mode, '--checkpoint', str(checkpoint), '--wav', str(wav)])
    assert app.module.main(argv) == 0
    assert option(app.calls[0][0], '--device') == 'cuda'


@pytest.mark.parametrize('model,directory', [('cnn', 'CNN1D-VAE'), ('lstm', 'LSTM-VAE')])
@pytest.mark.parametrize('task,script,output_flag', [
    ('reconstruction', 'run_demo.py', '--output-dir'),
    ('msm', 'train_msm.py', '--output'),
    ('classification', 'train_fourclass.py', '--output-dir'),
])
def test_model_and_workflow_mapping_uses_current_python(app, tmp_path, model, directory, task, script, output_flag):
    output = tmp_path/'outputs with spaces'
    argv = ['--model', model, '--output-dir', str(output)]
    if task != 'reconstruction':
        argv.extend(['--task', task])  # omission must select reconstruction
    assert app.module.main(argv) == 0
    assert len(app.calls) == 1
    command, kwargs = app.calls[0]
    assert command[0] == sys.executable
    assert Path(command[1]) == PROJECT/directory/script
    assert app.timestamp_inputs == [output.resolve()]
    assert Path(option(command, output_flag)) == app.training_roots[0]/task/model
    assert Path(kwargs['cwd']) == PROJECT
    assert kwargs['shell'] is False
    assert '--model' not in command and '--task' not in command


@pytest.mark.parametrize('task,output_flag', [('reconstruction', '--output-dir'), ('msm', '--output'), ('classification', '--output-dir')])
def test_both_is_cnn_then_lstm_with_separate_output_directories(app, tmp_path, task, output_flag):
    assert app.module.main(['--model', 'both', '--task', task, '--output-dir', str(tmp_path)]) == 0
    commands = [command for command, _ in app.calls]
    assert [Path(command[1]).parent.name for command in commands] == ['CNN1D-VAE', 'LSTM-VAE']
    assert len(app.training_roots) == 1
    assert [Path(option(command, output_flag)) for command in commands] == [
        app.training_roots[0]/task/'cnn', app.training_roots[0]/task/'lstm']


@pytest.mark.parametrize('separator', [False, True])
def test_csv_path_with_spaces_and_epochs_are_forwarded_as_individual_arguments(app, tmp_path, separator):
    split_csv = tmp_path/'split tables'/'my split.csv'
    argv = ['--model', 'both', '--output-dir', str(tmp_path/'runs')]
    if separator:
        argv.append('--')
    argv.extend(['--split-csv', str(split_csv), '--epochs', '2'])
    assert app.module.main(argv) == 0
    for command, kwargs in app.calls:
        assert option(command, '--split-csv') == str(split_csv)
        assert command.count(str(split_csv)) == 1
        assert option(command, '--epochs') == '2'
        assert '--' not in command
        assert isinstance(command, list) and kwargs['shell'] is False


def test_tasks_do_not_share_the_same_model_output_directory(app, tmp_path):
    app.module.main(['--model', 'cnn', '--output-dir', str(tmp_path)])
    app.module.main(['--model', 'cnn', '--task', 'msm', '--output-dir', str(tmp_path)])
    reconstruction = Path(option(app.calls[0][0], '--output-dir'))
    classification = Path(option(app.calls[1][0], '--output'))
    assert reconstruction == app.training_roots[0]/'reconstruction'/'cnn'
    assert classification == app.training_roots[1]/'msm'/'cnn'
    assert reconstruction != classification


def test_fourclass_alias_selects_configurable_classification(app, tmp_path):
    assert app.module.main(['--model', 'cnn', '--task', 'fourclass', '--output-dir', str(tmp_path)]) == 0
    command, _ = app.calls[0]
    assert Path(command[1]) == PROJECT/'CNN1D-VAE'/'train_fourclass.py'
    assert Path(option(command, '--output-dir')) == app.training_roots[0]/'classification'/'cnn'


def test_classification_both_pins_one_active_dataset_before_dispatch(app, tmp_path, monkeypatch):
    first = tmp_path/'first prepared version'
    second = tmp_path/'new prepared version'
    active = [first]
    resolutions = []
    original_dispatch = app.module.subprocess.run

    def resolve_active():
        resolutions.append(active[0])
        assert not app.calls  # Resolution must happen before either subprocess.
        return active[0]

    def change_active_after_first_model(command, **kwargs):
        result = original_dispatch(command, **kwargs)
        active[0] = second
        return result

    monkeypatch.setattr(app.module, 'resolve_dataset_dir', resolve_active)
    monkeypatch.setattr(app.module.subprocess, 'run', change_active_after_first_model)
    assert app.module.main(['--model', 'both', '--task', 'classification',
                            '--output-dir', str(tmp_path/'runs')]) == 0
    assert resolutions == [first] and active[0] == second
    assert len(app.calls) == 2
    for command, _ in app.calls:
        assert command.count('--data-dir') == 1
        assert option(command, '--data-dir') == str(first)


@pytest.mark.parametrize('equals_form', [False, True])
def test_explicit_classification_dataset_bypasses_active_resolution(app, tmp_path, monkeypatch, equals_form):
    explicit = tmp_path/'explicit prepared data'

    def unexpected_resolution():
        pytest.fail('Explicit --data-dir must not consult the active dataset pointer')

    monkeypatch.setattr(app.module, 'resolve_dataset_dir', unexpected_resolution)
    data_arguments = [f'--data-dir={explicit}'] if equals_form else ['--data-dir', str(explicit)]
    assert app.module.main(['--model', 'both', '--task', 'classification',
                            '--output-dir', str(tmp_path/'runs'), *data_arguments]) == 0
    assert len(app.calls) == 2
    for command, _ in app.calls:
        assert sum(value.split('=', 1)[0] == '--data-dir' for value in command) == 1
        if equals_form:
            assert f'--data-dir={explicit}' in command
        else:
            assert option(command, '--data-dir') == str(explicit)


def test_classification_resume_leaves_dataset_selection_to_checkpoint(app, tmp_path, monkeypatch):
    checkpoint = tmp_path/'saved last.pt'
    checkpoint.write_bytes(b'checkpoint fixture; never loaded')

    def unexpected_resolution():
        pytest.fail('Resume must not consult the active dataset pointer')

    monkeypatch.setattr(app.module, 'resolve_dataset_dir', unexpected_resolution)
    assert app.module.main(['--model', 'cnn', '--task', 'classification',
                            '--resume', str(checkpoint), '--output-dir', str(tmp_path/'runs')]) == 0
    command, _ = app.calls[0]
    assert option(command, '--resume') == str(checkpoint.resolve())
    assert not any(value.split('=', 1)[0] == '--data-dir' for value in command)


@pytest.mark.parametrize('equals_form', [False, True])
def test_combined_classification_csv_is_forwarded_to_both_without_active_dataset(app, tmp_path, monkeypatch, equals_form):
    selected = tmp_path/'chosen prepared version'/'samples.csv'

    def unexpected_resolution():
        pytest.fail('Explicit --csv must let the trainer infer the dataset directory')

    monkeypatch.setattr(app.module, 'resolve_dataset_dir', unexpected_resolution)
    csv_arguments = [f'--csv={selected}'] if equals_form else ['--csv', str(selected)]
    assert app.module.main(['--model', 'both', '--task', 'classification',
                            '--output-dir', str(tmp_path/'runs'), *csv_arguments]) == 0
    assert len(app.calls) == 2
    for command, _ in app.calls:
        assert not any(value.split('=', 1)[0] == '--data-dir' for value in command)
        assert sum(value.split('=', 1)[0] == '--csv' for value in command) == 1
        if equals_form:
            assert f'--csv={selected}' in command
        else:
            assert option(command, '--csv') == str(selected)


@pytest.mark.parametrize('returncodes,expected_count,expected_code', [([7, 0], 1, 7), ([0, 19], 2, 19)])
def test_both_stops_on_failure_and_returns_the_child_exit_code(app, tmp_path, returncodes, expected_count, expected_code):
    app.returncodes[:] = returncodes
    result = app.module.main(['--model', 'both', '--output-dir', str(tmp_path)])
    assert result == expected_code
    assert len(app.calls) == expected_count


def test_no_arguments_show_help_without_starting_training(app, capsys):
    assert app.module.main([]) == 0
    assert not app.calls
    assert '--model' in capsys.readouterr().out


@pytest.mark.parametrize('argv', [['--model', 'transformer'], ['--epochs', '1']])
def test_invalid_or_missing_model_is_rejected_before_training(app, argv):
    with pytest.raises(SystemExit) as raised:
        app.module.main(argv)
    assert raised.value.code == 2
    assert not app.calls


@pytest.mark.parametrize('extra', [
    ['--output', 'bypass'],
    ['--', '--output=bypass'],
    ['--', '--output-dir', 'bypass'],
])
def test_forwarded_output_flags_cannot_override_model_isolation(app, tmp_path, extra):
    with pytest.raises(SystemExit) as raised:
        app.module.main(['--model', 'both', '--output-dir', str(tmp_path), *extra])
    assert raised.value.code == 2
    assert not app.calls


def test_main_without_explicit_argv_reads_sys_argv(app, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['main.py', '--model', 'lstm', '--output-dir', str(tmp_path), '--epochs', '1'])
    assert app.module.main() == 0
    assert Path(app.calls[0][0][1]) == PROJECT/'LSTM-VAE'/'run_demo.py'
    assert option(app.calls[0][0], '--epochs') == '1'


def test_default_output_root_is_project_runs(app):
    assert app.module.main(['--model', 'cnn']) == 0
    assert app.timestamp_inputs == [PROJECT/'runs']
    assert Path(option(app.calls[0][0], '--output-dir')) == app.training_roots[0]/'reconstruction'/'cnn'


def test_repeated_training_calls_get_different_run_directories(app, tmp_path):
    output_root = tmp_path/'experiments'
    for _ in range(2):
        assert app.module.main(['--mode', 'train', '--model', 'cnn', '--output-dir', str(output_root)]) == 0
    destinations = [Path(option(command, '--output-dir')) for command, _ in app.calls]
    assert len(app.training_roots) == 2
    assert destinations[0] != destinations[1]
    assert destinations == [root/'reconstruction'/'cnn' for root in app.training_roots]
    assert not output_root.exists()  # neither mocked dispatch nor timestamp generation creates run artifacts


@pytest.mark.parametrize('model', ['cnn', 'lstm'])
@pytest.mark.parametrize('task,output_flag', [('reconstruction', '--output-dir'), ('msm', '--output'), ('classification', '--output-dir')])
def test_resume_forwards_checkpoint_and_total_epochs_to_a_new_run(app, tmp_path, monkeypatch, model, task, output_flag):
    checkpoint = tmp_path/'prior run'/'last.pt'
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'existing checkpoint fixture; trainer is not run')
    monkeypatch.setattr(app.module, 'PROJECT_ROOT', tmp_path)
    argv = ['--mode', 'train', '--model', model, '--task', task,
            '--resume', str(checkpoint.relative_to(tmp_path)), '--epochs', '300',
            '--eval-every', '20', '--output-dir', 'new runs']
    assert app.module.main(argv) == 0
    command, _ = app.calls[0]
    assert option(command, '--resume') == str(checkpoint.resolve())
    assert command.count('--resume') == 1
    assert option(command, '--epochs') == '300'
    assert option(command, '--eval-every') == '20'
    assert Path(option(command, output_flag)) == app.training_roots[0]/task/model
    assert checkpoint.read_bytes() == b'existing checkpoint fixture; trainer is not run'
    assert Path(option(command, output_flag)) != checkpoint.parent


@pytest.mark.parametrize('mode', ['predict', 'stream'])
def test_resume_is_rejected_in_nontraining_modes(app, prediction_files, mode):
    checkpoint, wav = prediction_files
    with pytest.raises(SystemExit) as raised:
        app.module.main(['--mode', mode, '--model', 'cnn', '--checkpoint', str(checkpoint),
                         '--wav', str(wav), '--resume', str(checkpoint)])
    assert raised.value.code == 2
    assert not app.calls and not app.training_roots


def test_resume_requires_one_model(app, tmp_path):
    checkpoint = tmp_path/'last.pt'
    checkpoint.write_bytes(b'checkpoint fixture')
    with pytest.raises(SystemExit) as raised:
        app.module.main(['--mode', 'train', '--model', 'both', '--resume', str(checkpoint)])
    assert raised.value.code == 2
    assert not app.calls and not app.training_roots


@pytest.mark.parametrize('is_directory', [False, True])
def test_resume_requires_an_existing_checkpoint_file(app, tmp_path, is_directory):
    checkpoint = tmp_path/'last.pt'
    if is_directory:
        checkpoint.mkdir()
    with pytest.raises(SystemExit) as raised:
        app.module.main(['--model', 'cnn', '--resume', str(checkpoint)])
    assert raised.value.code == 2
    assert not app.calls and not app.training_roots


def test_resume_cannot_bypass_launcher_checks_after_separator(app, tmp_path):
    checkpoint = tmp_path/'last.pt'
    checkpoint.write_bytes(b'checkpoint fixture')
    with pytest.raises(SystemExit) as raised:
        app.module.main(['--model', 'both', '--', '--resume', str(checkpoint)])
    assert raised.value.code == 2
    assert not app.calls


@pytest.fixture
def prediction_files(tmp_path):
    checkpoint = tmp_path/'checkpoint files'/'trained 200.pt'
    wav = tmp_path/'recordings'/'record 01'/'audio sample.wav'
    checkpoint.parent.mkdir(parents=True)
    wav.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'checkpoint fixture; never loaded')
    wav.write_bytes(b'WAV fixture; never decoded')
    return checkpoint, wav


@pytest.mark.parametrize('model', ['cnn', 'lstm'])
@pytest.mark.parametrize('input_mode', ['file', 'stream'])
def test_infer_defaults_to_classification_and_routes_selected_input(app, prediction_files, tmp_path, model, input_mode):
    from datetime import datetime

    checkpoint, wav = prediction_files
    output_root = tmp_path/'inference outputs'
    assert app.module.main(['--mode', 'infer', '--input-mode', input_mode, '--model', model,
                            '--checkpoint', str(checkpoint), '--wav', str(wav),
                            '--output-dir', str(output_root)]) == 0
    assert len(app.calls) == 1
    command, kwargs = app.calls[0]
    assert command[:2] == [sys.executable, str(PROJECT/'Integration'/'simulate_monitor.py')]
    assert option(command, '--mode') == 'infer'
    assert option(command, '--input-mode') == input_mode
    assert option(command, '--model') == model
    assert Path(option(command, '--checkpoint')) == checkpoint.resolve()
    assert Path(option(command, '--wav')) == wav.resolve()
    parts = Path(option(command, '--output-dir')).relative_to(output_root.resolve()).parts
    assert parts[0] == 'inference'
    datetime.strptime(parts[1], '%Y-%m-%d_%H-%M-%S_%f')
    assert parts[2:] == ('classification', model, input_mode)
    assert '--output' not in command and '--max-windows' not in command
    assert kwargs == {'cwd': PROJECT, 'shell': False}
    if input_mode == 'stream':
        assert '--realtime' in command
        assert float(option(command, '--chunk-ms')) == 20.0


def test_infer_defaults_to_file_and_forwards_cpu_threads(app, prediction_files, tmp_path):
    checkpoint, wav = prediction_files
    assert app.module.main(['--mode', 'infer', '--model', 'cnn',
                            '--checkpoint', str(checkpoint), '--wav', str(wav),
                            '--output-dir', str(tmp_path/'outputs'), '--threads', '2']) == 0
    command, _ = app.calls[0]
    assert option(command, '--input-mode') == 'file'
    assert option(command, '--threads') == '2'
    assert '--realtime' not in command and '--no-realtime' not in command


def test_infer_stream_can_run_without_realtime_pacing(app, prediction_files):
    checkpoint, wav = prediction_files
    assert app.module.main(['--mode', 'infer', '--input-mode', 'stream', '--model', 'cnn',
                            '--checkpoint', str(checkpoint), '--wav', str(wav),
                            '--chunk-ms', '12.5', '--no-realtime']) == 0
    command, _ = app.calls[0]
    assert '--no-realtime' in command and '--realtime' not in command
    assert float(option(command, '--chunk-ms')) == 12.5


def test_training_rejects_inference_input_mode_before_dispatch(app, tmp_path):
    with pytest.raises(SystemExit) as raised:
        app.module.main(['--mode', 'train', '--model', 'cnn', '--input-mode', 'file',
                         '--output-dir', str(tmp_path)])
    assert raised.value.code == 2
    assert not app.calls and not app.training_roots


@pytest.mark.parametrize('input_mode,script', [('file', 'predict.py'), ('stream', 'stream_wav.py')])
def test_infer_explicit_reconstruction_reuses_legacy_scripts(app, prediction_files, input_mode, script):
    checkpoint, wav = prediction_files
    assert app.module.main(['--mode', 'infer', '--input-mode', input_mode,
                            '--task', 'reconstruction', '--model', 'lstm',
                            '--checkpoint', str(checkpoint), '--wav', str(wav)]) == 0
    command, _ = app.calls[0]
    assert command[:2] == [sys.executable, str(PROJECT/'LSTM-VAE'/script)]
    assert '--input-mode' not in command and '--mode' not in command


@pytest.mark.parametrize('model,directory', [('cnn', 'CNN1D-VAE'), ('lstm', 'LSTM-VAE')])
def test_prediction_routes_to_matching_script_and_defaults_to_all_windows(app, prediction_files, tmp_path, model, directory):
    checkpoint, wav = prediction_files
    output_root = tmp_path/'prediction outputs'
    assert app.module.main(['--mode', 'predict', '--model', model,
                            '--checkpoint', str(checkpoint), '--wav', str(wav),
                            '--output-dir', str(output_root)]) == 0
    assert len(app.calls) == 1
    command, kwargs = app.calls[0]
    assert command[:2] == [sys.executable, str(PROJECT/directory/'predict.py')]
    assert option(command, '--checkpoint') == str(checkpoint.resolve())
    assert option(command, '--wav') == str(wav.resolve())
    assert option(command, '--max-windows') == '0'
    expected = output_root/'prediction'/model/wav.parent.name/f'{wav.stem}_{checkpoint.stem}.json'
    assert Path(option(command, '--output')) == expected
    assert kwargs == {'cwd': PROJECT, 'shell': False}
    assert '--mode' not in command and '--model' not in command
    assert not app.training_roots


def test_prediction_resolves_relative_paths_and_preserves_custom_output_and_extra_arguments(app, prediction_files, tmp_path, monkeypatch):
    checkpoint, wav = prediction_files
    monkeypatch.setattr(app.module, 'PROJECT_ROOT', tmp_path)
    output = Path('exact results')/'custom prediction.json'
    argv = ['--mode', 'predict', '--model', 'cnn',
            '--checkpoint', str(checkpoint.relative_to(tmp_path)),
            '--wav', str(wav.relative_to(tmp_path)), '--output', str(output),
            '--max-windows', '7', '--', '--batch-size', '4', '--threads', '2']
    assert app.module.main(argv) == 0
    command, kwargs = app.calls[0]
    assert Path(command[1]) == tmp_path/'CNN1D-VAE'/'predict.py'
    assert option(command, '--checkpoint') == str(checkpoint)
    assert option(command, '--wav') == str(wav)
    assert Path(option(command, '--output')) == tmp_path/output
    assert option(command, '--max-windows') == '7'
    assert option(command, '--batch-size') == '4'
    assert option(command, '--threads') == '2'
    assert '--' not in command
    assert kwargs['cwd'] == tmp_path and kwargs['shell'] is False


@pytest.mark.parametrize('model,directory', [('cnn', 'CNN1D-VAE'), ('lstm', 'LSTM-VAE')])
def test_fourclass_prediction_uses_one_existing_wav(app, prediction_files, tmp_path, model, directory):
    checkpoint, wav = prediction_files
    assert app.module.main(['--mode', 'predict', '--task', 'classification', '--model', model,
                            '--checkpoint', str(checkpoint), '--wav', str(wav),
                            '--output-dir', str(tmp_path)]) == 0
    command, _ = app.calls[0]
    assert Path(command[1]) == PROJECT/directory/'predict_fourclass.py'
    assert option(command, '--wav') == str(wav)
    assert '--max-windows' not in command
    assert Path(option(command, '--output')) == (
        tmp_path/'prediction'/'classification'/model/wav.parent.parent.name/wav.parent.name/f'{wav.stem}_{checkpoint.stem}.json'
    )


def test_fourclass_prediction_rejects_window_selection(app, prediction_files):
    checkpoint, wav = prediction_files
    with pytest.raises(SystemExit):
        app.module.main(['--mode', 'predict', '--task', 'classification', '--model', 'cnn',
                         '--checkpoint', str(checkpoint), '--wav', str(wav), '--max-windows', '1'])
    assert not app.calls


def test_classification_variants_have_distinct_default_outputs(app, prediction_files, tmp_path):
    checkpoint, _ = prediction_files
    for label in ('good', 'bad'):
        wav = tmp_path/'test'/label/'same_recording'/'segment_0000.wav'
        wav.parent.mkdir(parents=True)
        wav.write_bytes(b'fixture: never decoded')
        assert app.module.main(['--mode', 'predict', '--model', 'cnn', '--task', 'classification',
                                '--checkpoint', str(checkpoint), '--wav', str(wav)]) == 0
    destinations = [Path(option(command, '--output')) for command, _ in app.calls]
    assert destinations[0] != destinations[1]
    assert destinations[0].parents[1].name == 'good'
    assert destinations[1].parents[1].name == 'bad'


def test_prediction_expands_user_paths_without_changing_environment(app, prediction_files, tmp_path):
    checkpoint, wav = prediction_files
    output = tmp_path/'expanded output.json'
    def user_path(path):
        return '~/' + os.path.relpath(path, Path.home())
    assert app.module.main(['--mode', 'predict', '--model', 'cnn',
                            '--checkpoint', user_path(checkpoint), '--wav', user_path(wav),
                            '--output', user_path(output)]) == 0
    command, _ = app.calls[0]
    assert Path(option(command, '--checkpoint')) == checkpoint.resolve()
    assert Path(option(command, '--wav')) == wav.resolve()
    assert Path(option(command, '--output')) == output.resolve()


@pytest.mark.parametrize('missing', ['--checkpoint', '--wav'])
def test_prediction_requires_both_input_paths(app, prediction_files, missing):
    checkpoint, wav = prediction_files
    arguments = {'--checkpoint': str(checkpoint), '--wav': str(wav)}
    arguments.pop(missing)
    argv = ['--mode', 'predict', '--model', 'cnn']
    for key, value in arguments.items():
        argv.extend([key, value])
    with pytest.raises(SystemExit) as raised:
        app.module.main(argv)
    assert raised.value.code == 2
    assert not app.calls


@pytest.mark.parametrize('model,task', [('both', 'reconstruction'), ('cnn', 'msm')])
def test_prediction_rejects_unsupported_model_or_task(app, prediction_files, model, task):
    checkpoint, wav = prediction_files
    with pytest.raises(SystemExit) as raised:
        app.module.main(['--mode', 'predict', '--model', model, '--task', task,
                         '--checkpoint', str(checkpoint), '--wav', str(wav)])
    assert raised.value.code == 2
    assert not app.calls


def test_unknown_mode_is_rejected_without_subprocess(app):
    with pytest.raises(SystemExit) as raised:
        app.module.main(['--mode', 'evaluate', '--model', 'cnn'])
    assert raised.value.code == 2
    assert not app.calls


@pytest.mark.parametrize('invalid_input', ['missing_checkpoint', 'directory_wav'])
def test_prediction_requires_existing_regular_input_files(app, prediction_files, tmp_path, invalid_input):
    checkpoint, wav = prediction_files
    if invalid_input == 'missing_checkpoint':
        checkpoint = tmp_path/'missing.pt'
    else:
        wav = wav.parent
    with pytest.raises(SystemExit) as raised:
        app.module.main(['--mode', 'predict', '--model', 'cnn',
                         '--checkpoint', str(checkpoint), '--wav', str(wav)])
    assert raised.value.code == 2
    assert not app.calls


@pytest.mark.parametrize('overwrite', ['checkpoint', 'wav'])
def test_prediction_output_cannot_overwrite_either_source(app, prediction_files, tmp_path, monkeypatch, overwrite):
    checkpoint, wav = prediction_files
    monkeypatch.setattr(app.module, 'PROJECT_ROOT', tmp_path)
    source = checkpoint if overwrite == 'checkpoint' else wav
    original = source.read_bytes()
    with pytest.raises(SystemExit) as raised:
        app.module.main(['--mode', 'predict', '--model', 'cnn',
                         '--checkpoint', str(checkpoint), '--wav', str(wav),
                         '--output', str(source.relative_to(tmp_path))])
    assert raised.value.code == 2
    assert source.read_bytes() == original
    assert not app.calls


@pytest.mark.parametrize('option_name,value', [
    ('--checkpoint', 'weights.pt'), ('--wav', 'input.wav'),
    ('--output', 'result.json'), ('--max-windows', '0'),
])
def test_training_rejects_prediction_only_options(app, tmp_path, option_name, value):
    with pytest.raises(SystemExit) as raised:
        app.module.main(['--model', 'cnn', '--output-dir', str(tmp_path), option_name, value])
    assert raised.value.code == 2
    assert not app.calls


def test_prediction_returns_subprocess_failure(app, prediction_files):
    checkpoint, wav = prediction_files
    app.returncodes[0] = 23
    assert app.module.main(['--mode', 'predict', '--model', 'lstm',
                            '--checkpoint', str(checkpoint), '--wav', str(wav)]) == 23
    assert len(app.calls) == 1


def test_prediction_output_cannot_bypass_source_protection_after_separator(app, prediction_files):
    checkpoint, wav = prediction_files
    with pytest.raises(SystemExit) as raised:
        app.module.main(['--mode', 'predict', '--model', 'cnn',
                         '--checkpoint', str(checkpoint), '--wav', str(wav),
                         '--', '--output', str(checkpoint)])
    assert raised.value.code == 2
    assert not app.calls


def stream_arguments(checkpoint, wav, model='cnn'):
    return ['--mode', 'stream', '--model', model,
            '--checkpoint', str(checkpoint), '--wav', str(wav)]


@pytest.mark.parametrize('model,directory', [('cnn', 'CNN1D-VAE'), ('lstm', 'LSTM-VAE')])
def test_stream_routes_to_matching_script_with_realtime_and_default_chunk(app, prediction_files, tmp_path, capsys, model, directory):
    checkpoint, wav = prediction_files
    output_root = tmp_path/'stream outputs'
    assert app.module.main([*stream_arguments(checkpoint, wav, model), '--output-dir', str(output_root)]) == 0
    command, kwargs = app.calls[0]
    assert command[:2] == [sys.executable, str(PROJECT/directory/'stream_wav.py')]
    assert command.count('--realtime') == 1
    assert float(option(command, '--chunk-ms')) == 20.0
    assert '--max-windows' not in command and '--reference-json' not in command
    assert option(command, '--checkpoint') == str(checkpoint)
    assert option(command, '--wav') == str(wav)
    output = output_root/'stream'/model/wav.parent.name/f'{wav.stem}_{checkpoint.stem}.jsonl'
    assert Path(option(command, '--output')) == output
    assert Path(option(command, '--report')) == output.with_name(output.stem + '_report.json')
    assert kwargs == {'cwd': PROJECT, 'shell': False}
    assert not app.training_roots
    text = capsys.readouterr().out
    assert f'Model: {directory}' in text
    assert any(line.startswith('---') for line in text.splitlines())
    assert 'Processed' not in text
    assert str(PROJECT/directory/'stream_wav.py') not in text


@pytest.mark.parametrize('path_style', ['relative', 'user'])
def test_stream_custom_output_report_and_reference_paths_with_spaces(app, prediction_files, tmp_path, monkeypatch, path_style):
    checkpoint, wav = prediction_files
    reference = tmp_path/'offline comparison'/'reference results.json'
    reference.parent.mkdir()
    reference.write_text('{}')
    output = tmp_path/'live output'/'custom stream.jsonl'
    report = tmp_path/'report output'/'custom report.json'
    monkeypatch.setattr(app.module, 'PROJECT_ROOT', tmp_path)

    def argument(path):
        if path_style == 'user':
            return '~/' + os.path.relpath(path, Path.home())
        return str(path.relative_to(tmp_path))

    argv = stream_arguments(argument(checkpoint), argument(wav))
    argv += ['--output', argument(output), '--report', argument(report),
             '--reference-json', argument(reference), '--chunk-ms', '12.5',
             '--', '--batch-size', '4', '--threads', '1']
    assert app.module.main(argv) == 0
    command, kwargs = app.calls[0]
    for flag, path in [('--checkpoint', checkpoint), ('--wav', wav), ('--output', output),
                       ('--report', report), ('--reference-json', reference)]:
        assert Path(option(command, flag)) == path.resolve()
    assert float(option(command, '--chunk-ms')) == 12.5
    assert option(command, '--batch-size') == '4'
    assert option(command, '--threads') == '1'
    assert '--realtime' in command and '--' not in command
    assert kwargs['cwd'] == tmp_path and kwargs['shell'] is False


@pytest.mark.parametrize('chunk_ms', ['0', '-2.5', 'nan', 'inf', '-inf'])
def test_stream_rejects_nonpositive_or_nonfinite_chunk_sizes(app, prediction_files, chunk_ms):
    with pytest.raises(SystemExit) as raised:
        app.module.main([*stream_arguments(*prediction_files), f'--chunk-ms={chunk_ms}'])
    assert raised.value.code == 2
    assert not app.calls


@pytest.mark.parametrize('mode', ['train', 'predict'])
@pytest.mark.parametrize('flag,value', [('--chunk-ms', '20'), ('--report', 'report.json'), ('--reference-json', 'reference.json')])
def test_stream_options_are_rejected_by_other_modes(app, prediction_files, mode, flag, value):
    argv = ['--mode', mode, '--model', 'cnn']
    if mode == 'predict':
        checkpoint, wav = prediction_files
        argv += ['--checkpoint', str(checkpoint), '--wav', str(wav)]
    with pytest.raises(SystemExit) as raised:
        app.module.main([*argv, flag, value])
    assert raised.value.code == 2
    assert not app.calls


@pytest.mark.parametrize('extra', [['--model', 'both'], ['--task', 'msm'], ['--max-windows', '0']])
def test_stream_rejects_both_msm_and_window_limit(app, prediction_files, extra):
    with pytest.raises(SystemExit) as raised:
        app.module.main([*stream_arguments(*prediction_files), *extra])
    assert raised.value.code == 2
    assert not app.calls


@pytest.mark.parametrize('model,directory', [('cnn', 'CNN1D-VAE'), ('lstm', 'LSTM-VAE')])
@pytest.mark.parametrize('fast', [False, True])
def test_classification_stream_uses_matching_model_and_pacing(app, prediction_files, model, directory, fast):
    argv = [*stream_arguments(*prediction_files, model), '--task', 'classification']
    if fast:
        argv += ['--no-realtime']
    assert app.module.main(argv) == 0
    command, _ = app.calls[0]
    assert Path(command[1]) == PROJECT/directory/'stream_classification.py'
    assert ('--no-realtime' in command) is fast
    assert ('--realtime' in command) is not fast
    assert option(command, '--chunk-ms') == '20'
    assert '--reference-json' not in command


def test_classification_stream_does_not_accept_reconstruction_reference(app, prediction_files):
    with pytest.raises(SystemExit):
        app.module.main([*stream_arguments(*prediction_files), '--task', 'classification',
                         '--reference-json', 'reconstruction.json'])
    assert not app.calls


def test_reconstruction_fast_replay_omits_old_store_true_flag(app, prediction_files):
    assert app.module.main([*stream_arguments(*prediction_files), '--no-realtime']) == 0
    command, _ = app.calls[0]
    assert Path(command[1]).name == 'stream_wav.py'
    assert '--realtime' not in command and '--no-realtime' not in command


@pytest.mark.parametrize('missing', ['--checkpoint', '--wav'])
def test_stream_requires_checkpoint_and_wav(app, prediction_files, missing):
    argv = stream_arguments(*prediction_files)
    index = argv.index(missing)
    del argv[index:index + 2]
    with pytest.raises(SystemExit) as raised:
        app.module.main(argv)
    assert raised.value.code == 2
    assert not app.calls


@pytest.mark.parametrize('invalid_input', ['checkpoint', 'wav'])
def test_stream_inputs_must_be_files(app, prediction_files, tmp_path, invalid_input):
    checkpoint, wav = prediction_files
    if invalid_input == 'checkpoint':
        checkpoint = tmp_path/'missing checkpoint.pt'
    else:
        wav = wav.parent
    with pytest.raises(SystemExit) as raised:
        app.module.main(stream_arguments(checkpoint, wav))
    assert raised.value.code == 2
    assert not app.calls


@pytest.mark.parametrize('is_directory', [False, True])
def test_stream_reference_must_be_an_existing_file(app, prediction_files, tmp_path, is_directory):
    reference = tmp_path/'invalid reference'
    if is_directory:
        reference.mkdir()
    with pytest.raises(SystemExit) as raised:
        app.module.main([*stream_arguments(*prediction_files), '--reference-json', str(reference)])
    assert raised.value.code == 2
    assert not app.calls


@pytest.mark.parametrize('destination', ['--output', '--report'])
@pytest.mark.parametrize('source', ['checkpoint', 'wav', 'reference'])
def test_stream_outputs_cannot_overwrite_any_source(app, prediction_files, tmp_path, monkeypatch, destination, source):
    checkpoint, wav = prediction_files
    reference = tmp_path/'offline reference.json'
    reference.write_text('{"source":"untouched"}')
    sources = {'checkpoint': checkpoint, 'wav': wav, 'reference': reference}
    originals = {key: path.read_bytes() for key, path in sources.items()}
    monkeypatch.setattr(app.module, 'PROJECT_ROOT', tmp_path)
    argv = [*stream_arguments(checkpoint, wav), '--reference-json', str(reference),
            destination, str(sources[source].relative_to(tmp_path))]
    with pytest.raises(SystemExit) as raised:
        app.module.main(argv)
    assert raised.value.code == 2
    assert not app.calls
    assert {key: path.read_bytes() for key, path in sources.items()} == originals


def test_stream_output_and_report_must_be_distinct_after_path_resolution(app, prediction_files, tmp_path, monkeypatch):
    monkeypatch.setattr(app.module, 'PROJECT_ROOT', tmp_path)
    output = tmp_path/'same output.jsonl'
    with pytest.raises(SystemExit) as raised:
        app.module.main([*stream_arguments(*prediction_files), '--output', str(output),
                         '--report', output.name])
    assert raised.value.code == 2
    assert not app.calls


def test_stream_selected_checkpoint_is_forwarded_and_names_the_output_snapshot(app, prediction_files, tmp_path):
    trained, wav = prediction_files
    untrained = trained.with_name('untrained.pt')
    untrained.write_bytes(b'untrained checkpoint fixture; never loaded')
    for checkpoint in (untrained, trained):
        assert app.module.main([*stream_arguments(checkpoint, wav), '--output-dir', str(tmp_path/'runs')]) == 0
    outputs = []
    for (command, _), checkpoint in zip(app.calls, (untrained, trained)):
        assert option(command, '--checkpoint') == str(checkpoint)
        output = Path(option(command, '--output'))
        outputs.append(output)
        assert output.name == f'{wav.stem}_{checkpoint.stem}.jsonl'
        assert Path(option(command, '--report')) == output.with_name(output.stem + '_report.json')
    assert outputs[0] != outputs[1]


def test_stream_returns_child_failure(app, prediction_files):
    app.returncodes[0] = 29
    assert app.module.main(stream_arguments(*prediction_files)) == 29
    assert len(app.calls) == 1
