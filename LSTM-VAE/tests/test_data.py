"""Small synthetic checks; no real recordings are needed by these tests."""
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import (  # noqa: E402
    ChannelNormalizer,
    build_dataset,
    read_wav,
    select_window_starts,
)
from Dataset.Preprocessing.dataset_splits import SPLIT_NAMES, write_split_csv  # noqa: E402


def write_wav(path, pcm, sample_rate=44100):
    pcm = np.asarray(pcm, dtype="<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(pcm.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def split_row(recording_id, split, source_recording_id=None):
    return {"recording_id": recording_id,
            "source_recording_id": source_recording_id or recording_id, "split": split}


def write_recording(root, recording_id, base=0, source_folder=None):
    folder = root / recording_id
    folder.mkdir()
    metadata = {"recording": recording_id}
    if source_folder is not None:
        metadata["SourceRecording"] = {"Folder": source_folder}
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    pcm = np.column_stack((base + np.arange(43), base - np.arange(43))).astype(np.int16)
    write_wav(folder / "audio_000000.wav", pcm)
    return pcm


class DataTests(unittest.TestCase):
    def test_pcm_stereo_values_and_channel_order_are_preserved(self):
        expected = np.array([[-32768, 32767], [0, 17], [1234, -5678]], dtype=np.int16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            write_wav(path, expected)
            sample_rate, actual = read_wav(path)
        self.assertEqual(sample_rate, 44100)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(actual, expected)

    def test_wrong_sample_rate_or_channel_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.wav"
            for rate, channels in [(48000, 2), (44100, 1)]:
                with self.subTest(sample_rate=rate, channels=channels):
                    write_wav(path, np.zeros((4, channels)), sample_rate=rate)
                    with self.assertRaisesRegex(ValueError, "expected 44100 Hz"):
                        read_wav(path)
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(1)
                wav.setframerate(44100)
                wav.writeframes(bytes([128, 128] * 4))
            with self.assertRaisesRegex(ValueError, "PCM16"):
                read_wav(path)

    def test_window_grid_is_even_nonoverlapping_and_complete(self):
        starts = select_window_starts(83, sequence_length=8, max_windows_per_recording=4)
        np.testing.assert_array_equal(starts, [0, 24, 48, 72])
        self.assertTrue(np.all(starts % 8 == 0))
        self.assertTrue(np.all(np.diff(starts) >= 8))
        self.assertTrue(np.all(starts + 8 <= 83))
        all_complete = select_window_starts(83, 8, 16)
        np.testing.assert_array_equal(all_complete, np.arange(0, 80, 8))
        with self.assertRaisesRegex(ValueError, "positive"):
            select_window_starts(83, 8, 0)
        with self.assertRaisesRegex(ValueError, "fewer"):
            select_window_starts(7, 8, 16)

    def test_normalizer_roundtrip_constant_floor_and_json(self):
        train = np.array([[[0, 7], [2, 7]], [[4, 7], [6, 7]]], dtype=np.float32)
        normalizer = ChannelNormalizer.fit(train)
        np.testing.assert_allclose(normalizer.mean_pcm, [3.0, 7.0])
        np.testing.assert_allclose(normalizer.scale_pcm, [np.sqrt(5.0), 1.0])
        transformed = normalizer.transform(train)
        self.assertEqual(transformed.dtype, np.float32)
        recovered = normalizer.inverse(transformed)
        self.assertEqual(recovered.dtype, np.float32)
        np.testing.assert_allclose(recovered, train, atol=1e-6)
        restored = ChannelNormalizer.from_dict(json.loads(json.dumps(normalizer.to_dict())))
        np.testing.assert_array_equal(restored.transform(train), transformed)
        full_scale = ChannelNormalizer.full_scale()
        extremes = np.array([[-32768.0, 32767.0]], dtype=np.float32)
        np.testing.assert_array_equal(full_scale.inverse(full_scale.transform(extremes)), extremes)
        np.testing.assert_array_equal(full_scale.mean_pcm, [0, 0])
        np.testing.assert_array_equal(full_scale.scale_pcm, [32768, 32768])

    def test_csv_split_training_normalization_and_excluded_directories(self):
        rows = [split_row("train_a", "train"), split_row("train_b", "train"),
                split_row("val_a", "val"), split_row("test_a", "test"),
                split_row("excluded_sample", "exclude")]
        sets = {split: {row["recording_id"] for row in rows if row["split"] == split}
                for split in SPLIT_NAMES}
        expected_train = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i, row in enumerate(rows[:-1]):
                split = row["split"]
                base = i * 10 if split == "train" else (20000 if split == "val" else -20000)
                pcm = write_recording(root, row["recording_id"], base)
                if split == "train":
                    expected_train.extend([pcm[start:start + 8] for start in (0, 16, 32)])
            # Neither excluded nor unlisted directories need readable metadata/WAVs.
            extras = ["excluded_sample", "not_in_csv"]
            for name in extras:
                (root / name).mkdir()
            write_split_csv(root / "dataset_splits.csv", rows)
            dataset = build_dataset(root, sequence_length=8, max_windows_per_recording=3,
                                    split_csv=root / "dataset_splits.csv")
            self.assertEqual(dataset["ignored_recordings"], sorted(extras))
            self.assertEqual(dataset["split_rows"], rows)
            self.assertEqual(dataset["split_csv"], str((root / "dataset_splits.csv").resolve()))
            for split, count in [("train", 6), ("val", 3), ("test", 3)]:
                self.assertEqual(dataset[split]["x"].shape, (count, 8, 2))
                self.assertEqual(dataset[split]["x"].dtype, np.float32)
                self.assertEqual({w["recording_id"] for w in dataset[split]["windows"]}, sets[split])
                for item in dataset[split]["windows"]:
                    self.assertEqual(item["source_recording_id"], item["recording_id"])
                    self.assertEqual(item["start_s"], item["start_frame"] / 44100)
                    self.assertEqual(item["end_s"], (item["start_frame"] + 8) / 44100)
            self.assertEqual(len(dataset["manifest"]), 4)
            self.assertTrue(all(m["frame_count"] == 43 and m["window_count"] == 3 for m in dataset["manifest"]))
            self.assertTrue(all(m["duration_s"] == 43 / 44100 for m in dataset["manifest"]))
            normalizer = ChannelNormalizer.fit(dataset["train"]["x"])
            expected = np.asarray(expected_train, dtype=np.float64)
            np.testing.assert_allclose(normalizer.mean_pcm, expected.mean(axis=(0, 1)))
            np.testing.assert_allclose(normalizer.scale_pcm, expected.std(axis=(0, 1)))
            before = normalizer.to_dict()
            dataset["val"]["x"].fill(32767)
            dataset["test"]["x"].fill(-32768)
            self.assertEqual(ChannelNormalizer.fit(dataset["train"]["x"]).to_dict(), before)

    def test_second_build_reads_edited_csv_in_same_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [split_row("train_a", "train"), split_row("val_a", "val"),
                    split_row("test_a", "test"), split_row("new_sample", "exclude")]
            for index, row in enumerate(rows):
                write_recording(root, row["recording_id"], base=index*100)
            csv_path = root / "dataset_splits.csv"
            write_split_csv(csv_path, rows)
            first = build_dataset(root, sequence_length=8, max_windows_per_recording=1, split_csv=csv_path)
            rows[0]["split"] = "exclude"
            rows[3]["split"] = "train"
            write_split_csv(csv_path, rows)
            second = build_dataset(root, sequence_length=8, max_windows_per_recording=1, split_csv=csv_path)
            self.assertEqual(first["train"]["windows"][0]["recording_id"], "train_a")
            self.assertEqual(second["train"]["windows"][0]["recording_id"], "new_sample")
            self.assertEqual(first["ignored_recordings"], ["new_sample"])
            self.assertEqual(second["ignored_recordings"], ["train_a"])
            self.assertFalse(np.array_equal(first["train"]["x"], second["train"]["x"]))

    def test_explicit_csv_path_and_windows_source_basename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [split_row("clip_a", "train", "original_run"),
                    split_row("val_a", "val"), split_row("test_a", "test")]
            write_recording(root, "clip_a", source_folder=r"D:\\recordings\\original_run")
            write_recording(root, "val_a")
            write_recording(root, "test_a")
            csv_path = root / "custom_split.csv"
            write_split_csv(csv_path, rows)
            dataset = build_dataset(root, sequence_length=8, split_csv=csv_path)
            self.assertEqual(dataset["split_csv"], str(csv_path.resolve()))
            self.assertEqual(dataset["train"]["windows"][0]["source_recording_id"], "original_run")
            self.assertEqual(dataset["manifest"][0]["source_recording_id"], "original_run")

    def test_missing_csv_selected_directory_and_wav_are_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                build_dataset(root, split_csv=root / "dataset_splits.csv")
            rows = [split_row("train_a", "train"), split_row("val_a", "val"), split_row("test_a", "test")]
            write_split_csv(root / "dataset_splits.csv", rows)
            for row in rows[:-1]:
                write_recording(root, row["recording_id"])
            with self.assertRaisesRegex(ValueError, "missing recordings: test_a"):
                build_dataset(root, sequence_length=8, split_csv=root / "dataset_splits.csv")
            write_recording(root, "test_a")
            (root / "test_a" / "audio_000000.wav").unlink()
            with self.assertRaises(FileNotFoundError):
                build_dataset(root, sequence_length=8, split_csv=root / "dataset_splits.csv")

    def test_same_source_across_splits_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [split_row("clip_a", "train", "original_run"),
                    split_row("clip_b", "val", "original_run"), split_row("test_a", "test")]
            write_split_csv(root / "dataset_splits.csv", rows)
            with self.assertRaisesRegex(ValueError, "original_run.*appears in both train and val"):
                build_dataset(root, sequence_length=8, split_csv=root / "dataset_splits.csv")

    def test_forged_source_id_is_rejected_against_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [split_row("clip_a", "train", "fake_source"),
                    split_row("val_a", "val"), split_row("test_a", "test")]
            write_recording(root, "clip_a", source_folder=r"D:\\raw\\original_run")
            write_recording(root, "val_a")
            write_recording(root, "test_a")
            write_split_csv(root / "dataset_splits.csv", rows)
            with self.assertRaisesRegex(ValueError, "source_recording_id mismatch.*metadata identifies 'original_run'"):
                build_dataset(root, sequence_length=8, split_csv=root / "dataset_splits.csv")


if __name__ == "__main__":
    unittest.main()
