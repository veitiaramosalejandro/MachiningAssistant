# WAV classification validation

Run from the project root after a training run has completed. Use its validation-selected `best.pt`; the current classifier reads one full five-second WAV per sample.

## Held-out CSV and streaming consistency

```bash
python Validation/verify_classification.py --model cnn --checkpoint "runs/train/<timestamp>/classification/cnn/best.pt" --output-dir "runs/verification/<new-name>/cnn_test"
```

This reads the dataset saved in the checkpoint, checks its identities and labels against the saved experiment, and verifies that no original source appears in checkpoint training data. It evaluates every test sample offline, then feeds that same sample in 20 ms chunks through a fresh streaming session. The checkpoint normalization and class mapping are reused throughout.

For the normal training command with `--csv Dataset/Processed_5s/samples.csv`, verification rereads that saved combined CSV and selects its `split=test` rows. It does not substitute the companion `test.csv`. Legacy runs without `--csv` use their saved dataset's split CSVs.

Preprocessing replaces the fixed `Dataset/Processed_5s/` directory after a successful build. If its inputs or targets have changed since training, identity validation rejects the mismatch. Old checkpoints remain available, but their historical test data is not automatically retained. Keep a separate copy before rebuilding when you need to preserve an older experiment. `--data-dir` changes the WAV root; the saved combined CSV path still has to contain the matching experiment rows. Train a new model for a changed dataset.

Outputs are `verification.json` (metrics and file/stream agreement), `test_predictions.csv` (each target, prediction and class probability), `stream_predictions.jsonl`, and `labels.json`. The `original_good_subset` field separately reports the current `good` class. The output directory must be new or empty. Use `--data-dir` only with a matching dataset and `--chunk-ms` to choose another input chunk duration.

The test summary prints **ACC / P (macro) / R (macro) / F1 (macro)**. The saved keys are `accuracy`, `macro_precision`, `macro_recall`, and `macro_f1` (0–1 values). P and R average the per-class precision and recall equally over all configured classes; a zero denominator contributes 0. Per-class details are retained under `per_class`. These metrics require the CSV ground-truth labels; an unlabeled live stream returns class scores, not test accuracy/precision/recall.

The current four labels describe artificially added sine-amplitude levels. Accuracy on this dataset is not an estimate of real tool-break or machining-quality detection performance.

## Replay one WAV as streamed input

```bash
python main.py --mode stream --task classification --model cnn --checkpoint "runs/train/<timestamp>/classification/cnn/best.pt" --wav "Dataset/Data/20261012_124615/audio_000000.wav" --output "runs/verification/<new-name>/cnn_stream.jsonl"
```

The launcher replays at recorded timing by default. Add `--no-realtime` for a fast engineering check. One classification is emitted after each complete five-second window; 20 ms is the delivery block size, not the prediction window length. The final incomplete window is counted in the report and is not silently padded. Each WAV starts a new session, and model state does not carry between windows.

Replace `cnn` with `lstm` only when using a matching LSTM classification checkpoint. A reconstructed-waveform checkpoint or the older three-class MSM weights cannot substitute for this task. Untrained weights can test interface behavior, but their classification accuracy should not be reported as a trained result.

New LSTM classification checkpoints may use `model_config.frame_size=441`: all five-second PCM values are grouped into 500 recurrent steps internally, and the external WAV/PCM window remains 220500-by-2. Both file and streaming prediction load the saved frame size. Older LSTM checkpoints without this field retain the original frame size of 1. The grouping changes model architecture, so speed measurements or compatibility checks are separate from trained classification accuracy.

The shared `WavClassifier.predict_pcm()` and `StreamingClassifier.push()` APIs provide the same preprocessing and inference behavior for future gRPC integration. WAV replay does not test a live microphone, independent-device synchronization, or company-side C# transport. See `Integration/README.md` for the interface study and draft contract.
