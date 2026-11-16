# MSM classification workflow

This directory provides the shared preprocessing and training code used by both model directories. The saved classifiers and their historical training/evaluation artifacts are located under `CNN1D-VAE/artifacts/msm_classification/` and `LSTM-VAE/artifacts/msm_classification/`.

## Data and input contract

The original MSM `imi_vm20i` subset contains 37 experiments. The training pipeline aligns Sound0 audio at 48 kHz with Acc0 spindle acceleration resampled from 51.2 kHz. Sound0 retains PCM counts; Acc0 uses mm/s². It selects 2,048-sample windows inside labeled cutting intervals and fits normalization only on training windows.

Class meanings:

- `0`: normal cutting.
- `1`: abnormal cutting requiring intervention.
- `2`: a pre-existing defective tool cutting without an abnormal cutting condition. Abnormal cutting with a defective tool is labeled `1`.

`../Dataset/Preprocessing/msm_dataset_splits.csv` is the editable recording-level assignment table. Its initial allocation is 23 training, 7 validation, and 7 test experiments. Every training invocation reads the current CSV from disk; `splits.json` and the split counts in `summary.json` inside the prepared cache are historical preprocessing artifacts, not training assignments. The company labels `machining`, `tool_measurement`, `tool_change`, and numeric quality ratings belong to a different task and are not mapped to these classes.

The CSV columns are `recording_id,source_recording_id,split`. For MSM, both IDs must be the same real experiment ID, such as `dataset12`; do not invent source groups. Set `split` to `train`, `val`, `test`, or `exclude`. An excluded recording, or a cached recording omitted from the CSV, does not participate in training, normalization, validation, or testing. Each selected split must still contain all three classes. Unknown recording IDs and duplicate or overlapping assignments are rejected.

## Prepare and train

Run from the project root. Preprocessing code and split tables live in `Dataset/Preprocessing/`; company recordings live in `Dataset/Data/`, which also contains the default MSM cache location `msm_prepared/`. The original approximately 36 GB MSM subset remains external: set `--raw` to its existing `imi_vm20i` directory. It is not copied into the company data directory, and no absolute dataset location is embedded in the code.

```bash
python Dataset/Preprocessing/prepare_msm.py --raw /path/to/MSM/imi_vm20i --output Dataset/Data/msm_prepared
python CNN1D-VAE/train_msm.py --prepared Dataset/Data/msm_prepared --split-csv Dataset/Preprocessing/msm_dataset_splits.csv --epochs 200
python LSTM-VAE/train_msm.py --prepared Dataset/Data/msm_prepared --split-csv Dataset/Preprocessing/msm_dataset_splits.csv --epochs 200
```

Existing prepared arrays can be reused in place by passing their directory directly to `--prepared`. Editing the split CSV does not require resampling or rebuilding those arrays. Preprocessing output, training input, and report input default to `Dataset/Data/msm_prepared`; the trainer's default split CSV is `Dataset/Preprocessing/msm_dataset_splits.csv`. No full raw MSM data copy is required. The preprocessing script exists only under `Dataset/Preprocessing/`, not as a second copy in this directory.

Each output directory contains `dataset_splits_used.csv`, a snapshot of the CSV actually read, and `data_split.json`, the actual train/validation/test recording lists. `training_config.json` and `classification_report.json` record the CSV source, excluded and unlisted recordings, ignored window count, and actual source assignments. Fresh runs fit normalization only on their selected training windows; full resumes reuse the saved normalization. Freeze the assignments before comparing models; do not repeatedly change the test split based on model results.

## Run directories, evaluation, and resume

Omitting `--output` creates a new `runs/train/<Europe-Lisbon-timestamp>/msm/<cnn-or-lstm>/` directory. An explicitly selected output directory must be empty, including when resuming; old results are not overwritten. The unified `main.py` entry point also supplies a unique training directory.

`--eval-every` defaults to 20. Validation and test metrics are observed at those epochs and at the final requested epoch. `--no-test-during-training` disables periodic test observations; the final evaluation still tests the validation-selected weights. Only validation interval macro F1 chooses the best checkpoint and controls early stopping. `--patience 7` means seven validation evaluation checks without improvement, not seven training epochs. History stores `null` for metrics not evaluated at an epoch; report plots use only observed points.

Saved files:

- `last.pt`: atomic state after the latest complete epoch; initialization is recorded as epoch 0.
- `checkpoints/epoch_0020.pt`, etc.: full states at periodic/final evaluation points.
- `best.pt` and `best_classifier.pt`: the validation-selected weights and their corresponding full training state.
- `final.pt`: the last completed epoch's weights and full state. Final report metrics continue to use the **best** weights, which can differ from the final weights.

Continue a run in a new timestamped output directory:

```bash
python CNN1D-VAE/train_msm.py --resume runs/train/<timestamp>/msm/cnn/last.pt --epochs 300
```

`--epochs` is the total target, so a checkpoint at epoch 200 continues with epoch 201. Unspecified paths and parameters come from the checkpoint. Full states contain the AdamW optimizer, Python/NumPy/PyTorch RNG states, every DataLoader generator, ordered history, best weights and selection state, normalization, preprocessing, and ordered prepared-window metadata/labels/shape. The current CSV is read again: active source assignments and ordered active window identities must match. Signal values are not rehashed. A full resume reuses its saved normalization and rejects changes to batch size, learning rate, loss weights, seed, CPU threads, and patience. The total epoch target may be extended; evaluation cadence and test observation may be changed explicitly and are recorded in the resume history.

RNG states are restored after objects have been constructed and loaded. An interrupted partial epoch is discarded: Ctrl+C exits with code 130 and retains the last complete-epoch checkpoint. Extending a run that already reached its former target retains that run's final evaluation point; use identical evaluation points when checking equivalence with an uninterrupted run.

A legacy `best_classifier.pt` without `training_state` supports **weights-only continuation** from `best_epoch`: the optimizer and RNG sequence are rebuilt, and the console explicitly states that this is not exact resume. Supply `--prepared` and `--split-csv` when those paths were not saved in the legacy file. Its history begins with a newly evaluated validation anchor; no earlier training losses are fabricated.

`experiment.py` contains the shared supervised trainer, validation-based checkpoint selection, window metrics, and labeled-interval metrics. `make_report.py` renders existing evaluation results; it does not train models. For example, after a new run:

```bash
python msm_classification/make_report.py --cnn-dir /path/to/completed/cnn --lstm-dir /path/to/completed/lstm --prepared Dataset/Data/msm_prepared --output runs/msm_report
```

The report reads actual assignments from each run's `data_split.json` and, when present, checks its saved CSV snapshot. Both models must have the same source-recording IDs in each split, not merely equal split sizes. Training/validation/test counts come from the run artifacts. The prepared summary is used only to describe the full preprocessing cache.

`data_comparison.json` is the historical source comparison for MSM and the original 16 company recordings. It is not an inventory of the current 31-sample company collection.

## Classifier inference

`StateEngine` in each model directory loads that model's `best_classifier.pt`. It accepts floating-point arrays of shape `(batch, 2048, 2)` or `(2048, 2)` with the same units and alignment as training. Softmax outputs are uncalibrated class scores; these checkpoints have not been validated for company deployment.

`probe_at16.py` preserves the numerical compatibility diagnostic on the original 16 company recordings. It reads those 16 IDs from the selected model directory's historical `artifacts/data_manifest.json`; it does not follow either editable training CSV. Its default source is the project's `Dataset/Data/`, from which only those historical IDs are selected. A missing, duplicate, or non-16-recording manifest is rejected rather than silently changing the diagnostic's scope. Keep this historical manifest when saving new runs to their own output directories. The probe does not provide formal company classification accuracy or infer a vibration calibration.

```bash
python msm_classification/probe_at16.py --model-dir CNN1D-VAE --output runs/cnn_at16_probe
```

## Sources

- [MSM dataset](https://www.kaggle.com/datasets/manufuturetoday/multi-sensor-for-metal-milling-anomaly), version 2, CC BY 4.0.
- [Original data descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC13314991/).
