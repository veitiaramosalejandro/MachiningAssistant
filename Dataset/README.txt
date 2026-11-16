Dataset
=======

After adding recording folders to Data/, run:

    python Dataset/Preprocessing/prepare_dataset.py

The preprocessor uses the existing Preprocessing/dataset_splits.csv source table.
It preserves current source assignments: a new clip inherits its original source's
split, while a previously unknown source defaults to train. Keep the supplied
SourceRecording.Folder metadata so same-source clips remain together.

Normal output is only the resulting combined samples.csv path. Pass that one
CSV to the existing classification training command:

    python main.py --mode train --task classification --model lstm --csv Dataset/Processed_5s/samples.csv --epochs 20

The split column supplies train/validation/test. Training is a separate manual step.
Each preparation builds and validates temporary output, then replaces the fixed
Processed_5s/ directory and updates Preprocessing/current_dataset.json.
Raw recordings and model weights are unchanged. The previous generated output is
replaced; keep a copy elsewhere if an older experiment needs it.
Use --dry-run for a read-only preview and --verbose for build progress.

Data/ stores only original recording folders and their supplied annotation files.
Processed_5s/ stores the generated WAVs as split/label/recording/segment.wav,
samples.csv, companion split CSVs, labels and generation/source snapshots.
segments.csv/json refer directly to the original good WAVs as base signals,
without a duplicate base-WAV set. base_dataset_summary.json records their counts.
Short tails remain in tails/ with tails.csv/json and their actual lengths.
The current 31 raw recordings yield 1476 complete five-second WAVs: 1180 train,
140 validation and 156 test. Future builds report their actual counts.
Train these WAVs manually from the project root:

    python main.py --mode train --task classification --model cnn --csv Dataset/Processed_5s/samples.csv --epochs 200
    python main.py --mode train --task classification --model lstm --csv Dataset/Processed_5s/samples.csv --epochs 200

Preprocessing/fourclass_data.py reads the CSVs and WAVs, normalizes using training
statistics, and prepares the entire five-second waveform for either model.
Fresh LSTM training defaults to frame_size=441: all 220500 stereo frames are
preserved and grouped into 500 recurrent steps. CNN pads 44 normalized-zero frames.
Rebuilt inputs that differ from a checkpoint's saved data identity require a new
training run; old weights are retained, but full resume rejects changed inputs.

Preprocessing/ stores the shared data-loading, windowing, normalization and split-reading code, editable split CSVs, MSM signal preparation, and the WAV-label demonstration script.

CNN1D-VAE and LSTM-VAE share Preprocessing/. Classification reads Processed_5s/;
the original reconstruction workflow reads Data/. Model definitions, training,
inference and saved weights remain outside these data directories.
