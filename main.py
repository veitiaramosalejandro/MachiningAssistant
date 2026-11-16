"""Train a model or run inference on a complete WAV or simulated real-time stream."""
from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

from training_support import timestamped_training_root
from Dataset.Preprocessing.active_dataset import resolve_dataset_dir


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIRECTORIES = {"cnn": "CNN1D-VAE", "lstm": "LSTM-VAE"}
WORKFLOWS = {
    "reconstruction": ("run_demo.py", "--output-dir", "WAV reconstruction training"),
    "msm": ("train_msm.py", "--output", "MSM three-class classification training"),
    "classification": ("train_fourclass.py", "--output-dir", "5-second WAV classification training"),
    "fourclass": ("train_fourclass.py", "--output-dir", "5-second WAV classification training"),
}


def project_path(value: Path) -> Path:
    value = value.expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def run_command(command: list[str], model_directory: str) -> int:
    try:
        result = subprocess.run(command, cwd=PROJECT_ROOT, shell=False)
    except KeyboardInterrupt:
        print('Run interrupted.', file=sys.stderr)
        return 130
    if result.returncode:
        print(f"{model_directory} failed with exit code {result.returncode}.", file=sys.stderr)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Other options are forwarded to the selected training or prediction script.\n"
            "Relative paths are interpreted from this project root.\n\n"
            "Examples:\n"
            "  python main.py --mode train --task classification --model cnn --epochs 200\n"
            "  python main.py --mode train --task classification --model lstm --epochs 200\n"
            "  python main.py --mode train --task classification --model lstm --csv /path/to/samples.csv --epochs 20\n"
            "  python main.py --mode infer --input-mode file --model cnn --checkpoint /path/to/best.pt --wav /path/to/recording.wav\n"
            "  python main.py --mode infer --input-mode stream --model lstm --checkpoint /path/to/best.pt --wav /path/to/recording.wav\n"
            "  python main.py --mode train --model cnn --epochs 3\n"
            "  python main.py --mode train --model both --epochs 3\n"
            "  python main.py --mode train --model cnn --epochs 200 --eval-every 20\n"
            "  python main.py --mode train --model cnn --resume /path/to/last.pt --epochs 300\n"
            "  python main.py --mode predict --model cnn --checkpoint runs/reconstruction/cnn/smoke_trained.pt --wav Dataset/Data/20261012_124615/audio_000000.wav\n"
            "  python main.py --mode stream --model cnn --checkpoint runs/reconstruction/cnn/smoke_trained.pt --wav Dataset/Data/20261012_124615/audio_000000.wav --chunk-ms 20\n"
            "  python main.py --mode train --model cnn --task msm --prepared /path/to/prepared --epochs 30\n\n"
            "In PyCharm, run main.py and enter the desired mode, model, and file options in Parameters."
        ),
    )
    parser.add_argument("--mode", choices=["train", "infer", "predict", "stream"], default="train",
                        help="train: fit weights (default); infer: predict with file or stream input; predict/stream: legacy commands")
    parser.add_argument("--input-mode", choices=["file", "stream"],
                        help="Infer only: file processes all complete WAV windows without pacing (default); stream simulates real-time input")
    parser.add_argument("--model", type=str.lower, choices=["cnn", "lstm", "both"], required=True,
                        help="Model to use; both runs CNN and then LSTM in training mode")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"],
                        help="Default auto: probe CUDA and use it when usable, otherwise CPU; optionally force cpu or cuda")
    parser.add_argument("--task", choices=list(WORKFLOWS),
                        help="Default: classification for infer, reconstruction otherwise. classification: 5-second WAV classes; fourclass: alias; reconstruction: WAV VAE; msm: MSM classification")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs",
                        help="Output root; training uses train/<timestamp>/<task>/<model>; classification infer uses inference/<timestamp>/classification/<model>/<input-mode>")
    parser.add_argument("--resume", type=Path,
                        help="Train only: resume one model from a checkpoint into a new timestamped directory")
    parser.add_argument("--checkpoint", type=Path,
                        help="Inference: checkpoint matching the selected model and task")
    parser.add_argument("--wav", type=Path,
                        help="Inference: 44.1 kHz stereo PCM16 WAV to process")
    parser.add_argument("--output", type=Path,
                        help="Legacy prediction/reconstruction: exact output JSON/JSONL path; classification infer uses --output-dir")
    parser.add_argument("--max-windows", type=int,
                        help="Prediction only: maximum complete windows; default 0 processes the entire WAV")
    parser.add_argument("--chunk-ms", type=float,
                        help="Stream only: incoming chunk duration in milliseconds (default 20)")
    parser.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=None,
                        help="Stream only: replay at recorded timing (default); --no-realtime runs quickly")
    parser.add_argument("--report", type=Path,
                        help="Stream only: summary JSON; default <output-stem>_report.json")
    parser.add_argument("--reference-json", type=Path,
                        help="Stream only: optional full offline prediction JSON for the same WAV and checkpoint")
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        parser.print_help()
        return 0
    args, extra_arguments = parser.parse_known_args(arguments)
    infer_requested = args.mode == 'infer'
    if args.input_mode is not None and not infer_requested:
        parser.error('--input-mode requires --mode infer.')
    if args.task is None:
        args.task = 'classification' if infer_requested else 'reconstruction'
    if args.task == "fourclass":
        args.task = "classification"
    if infer_requested:
        args.input_mode = args.input_mode or 'file'
        args.mode = 'stream' if args.input_mode == 'stream' else 'predict'
    if extra_arguments and extra_arguments[0] == "--":
        extra_arguments = extra_arguments[1:]
    reserved = {"--mode", "--input-mode", "--model", "--device", "--task", "--output", "--output-dir", "--checkpoint", "--wav", "--max-windows",
                "--chunk-ms", "--report", "--reference-json", "--resume", "--realtime", "--no-realtime"}
    if any(value.split("=", 1)[0] in reserved for value in extra_arguments):
        parser.error("Place mode, model, checkpoint, WAV, and output options before the -- separator.")
    if args.device is not None:
        extra_arguments.extend(['--device', args.device])

    output_root = project_path(args.output_dir)
    if args.mode != 'train' and args.resume is not None:
        parser.error('--resume requires --mode train.')
    if args.mode != "stream" and any(value is not None for value in (args.chunk_ms, args.report, args.reference_json, args.realtime)):
        parser.error("--chunk-ms, --report, --reference-json, and --realtime/--no-realtime require --mode infer --input-mode stream (or legacy --mode stream).")
    if args.mode in ("predict", "stream"):
        if args.model == "both":
            parser.error("Prediction requires one model (cnn or lstm) and its matching checkpoint.")
        if args.task == "msm":
            parser.error("MSM classification requires prepared Sound0/Acc0 input.")
        if args.task == "classification" and args.reference_json is not None:
            parser.error("--reference-json uses the reconstruction JSON format; omit it for classification.")
        if args.task == "classification" and args.max_windows is not None:
            parser.error("Classification processes complete 5-second windows; omit --max-windows.")
        if args.checkpoint is None or args.wav is None:
            parser.error("Prediction requires both --checkpoint and --wav.")
        if args.max_windows is not None and args.max_windows < 0:
            parser.error("--max-windows must be nonnegative; use 0 for all complete windows.")
        if args.mode == "stream" and args.max_windows is not None:
            parser.error("--max-windows applies to predict mode. Stream replay reads the entire WAV.")
        checkpoint, wav = project_path(args.checkpoint), project_path(args.wav)
        for label, path in [("Checkpoint", checkpoint), ("WAV", wav)]:
            if not path.is_file():
                parser.error(f"{label} file not found: {path}")
        streaming = args.mode == "stream"
        if infer_requested and args.task == 'classification':
            if args.output is not None or args.report is not None:
                parser.error('Classification inference saves responses.jsonl and report.json; choose their output root with --output-dir.')
            chunk_ms = args.chunk_ms if args.chunk_ms is not None else 20.0
            if not math.isfinite(chunk_ms) or chunk_ms <= 0:
                parser.error('--chunk-ms must be a finite positive number.')
            stamp = datetime.now(ZoneInfo('Europe/Lisbon')).strftime('%Y-%m-%d_%H-%M-%S_%f')
            output = output_root / 'inference' / stamp / 'classification' / args.model / args.input_mode
            command = [sys.executable, str(PROJECT_ROOT / 'Integration' / 'simulate_monitor.py'),
                       *extra_arguments, '--mode', 'infer', '--input-mode', args.input_mode,
                       '--model', args.model, '--checkpoint', str(checkpoint), '--wav', str(wav),
                       '--output-dir', str(output)]
            if streaming:
                command.extend(['--chunk-ms', f'{chunk_ms:g}',
                                '--no-realtime' if args.realtime is False else '--realtime'])
            print(f'Checkpoint: {checkpoint}', flush=True)
            print(f'WAV: {wav}', flush=True)
            return run_command(command, MODEL_DIRECTORIES[args.model])
        prediction_root = output_root / ("stream" if streaming else "prediction")
        if args.task == "classification":
            # Generated classes share recording directories and segment names.
            # Retain their enclosing class directory in output paths only.
            prediction_root = prediction_root / "classification" / args.model / wav.parent.parent.name
        else:
            prediction_root = prediction_root / args.model
        output = project_path(args.output) if args.output is not None else (
            prediction_root / wav.parent.name /
            f"{wav.stem}_{checkpoint.stem}.{'jsonl' if streaming else 'json'}"
        )
        if output in (checkpoint, wav):
            parser.error("The output JSON must not overwrite the checkpoint or input WAV.")
        model_directory = MODEL_DIRECTORIES[args.model]
        if streaming:
            chunk_ms = args.chunk_ms if args.chunk_ms is not None else 20.0
            if not math.isfinite(chunk_ms) or chunk_ms <= 0:
                parser.error("--chunk-ms must be a finite positive number.")
            report = project_path(args.report) if args.report is not None else output.with_name(f"{output.stem}_report.json")
            reference = project_path(args.reference_json) if args.reference_json is not None else None
            if reference is not None and not reference.is_file():
                parser.error(f"Offline reference JSON not found: {reference}")
            if report in (checkpoint, wav, output) or (reference is not None and reference in (output, report)):
                parser.error("Stream output and report must be distinct and must not overwrite input files.")
            stream_script = "stream_classification.py" if args.task == "classification" else "stream_wav.py"
            realtime = args.realtime is not False
            pacing = ["--realtime"] if realtime else (["--no-realtime"] if args.task == "classification" else [])
            command = [sys.executable, str(PROJECT_ROOT / model_directory / stream_script),
                       *extra_arguments, "--checkpoint", str(checkpoint), "--wav", str(wav),
                       "--chunk-ms", f"{chunk_ms:g}", *pacing, "--output", str(output), "--report", str(report)]
            if reference is not None:
                command.extend(["--reference-json", str(reference)])
            task_name = "WAV streaming classification" if args.task == "classification" else "WAV streaming prediction"
        elif args.task == "classification":
            command = [sys.executable, str(PROJECT_ROOT / model_directory / "predict_fourclass.py"),
                       *extra_arguments, "--checkpoint", str(checkpoint), "--wav", str(wav),
                       "--output", str(output)]
            task_name = "5-second WAV classification prediction"
        else:
            command = [sys.executable, str(PROJECT_ROOT / model_directory / "predict.py"),
                       *extra_arguments, "--checkpoint", str(checkpoint), "--wav", str(wav),
                       "--max-windows", str(args.max_windows if args.max_windows is not None else 0),
                       "--output", str(output)]
            task_name = "WAV reconstruction prediction"
        print('-' * 64, flush=True)
        if infer_requested:
            print(f'Mode: infer; Input: {args.input_mode}', flush=True)
        print(f"Model: {model_directory}; Task: {task_name}", flush=True)
        print(f"Checkpoint: {checkpoint}", flush=True)
        print(f"WAV: {wav}", flush=True)
        if streaming:
            print(f"Replay: {'real-time' if realtime else 'fast'} WAV input; Chunk size: {chunk_ms:g} ms", flush=True)
        print(f"Output: {output}", flush=True)
        if streaming:
            print(f"Report: {report}", flush=True)
        return run_command(command, model_directory)

    if any(value is not None for value in (args.checkpoint, args.wav, args.output, args.max_windows)):
        parser.error("--checkpoint, --wav, and --output require inference; --max-windows requires file reconstruction prediction. Use --output-dir for training results.")
    resume = project_path(args.resume) if args.resume is not None else None
    if resume is not None:
        if args.model == 'both':
            parser.error('--resume requires one model (cnn or lstm) and its matching checkpoint.')
        if not resume.is_file():
            parser.error(f'Resume checkpoint not found: {resume}')
    if (args.task == 'classification' and resume is None
            and not any(value.split('=', 1)[0] in ('--data-dir', '--csv') for value in extra_arguments)):
        # Pin one prepared version for this invocation, including --model both.
        extra_arguments = [*extra_arguments, '--data-dir', str(resolve_dataset_dir())]
    session = timestamped_training_root(output_root)
    selected = ["cnn", "lstm"] if args.model == "both" else [args.model]
    script_name, output_option, task_name = WORKFLOWS[args.task]

    for model in selected:
        model_directory = MODEL_DIRECTORIES[model]
        script = PROJECT_ROOT / model_directory / script_name
        output = session / args.task / model
        command = [sys.executable, str(script), *extra_arguments, output_option, str(output)]
        if resume is not None:
            command.extend(['--resume', str(resume)])
        print(f"Model: {model_directory}; Task: {task_name}", flush=True)
        print(f"Output: {output}", flush=True)
        if resume is not None:
            print(f'Resume: {resume}', flush=True)
        returncode = run_command(command, model_directory)
        if returncode:
            if returncode == 130 and (output/'last.pt').is_file():
                print(f'Resume checkpoint: {output / "last.pt"}', flush=True)
            return returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
