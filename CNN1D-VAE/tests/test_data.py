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
from Dataset.Preprocessing.dataset_splits import SPLIT_NAMES, write_split_csv


def write_wav(path, pcm, sample_rate=44100):
    pcm = np.asarray(pcm, dtype="<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(pcm.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def assignment(recording_id, split, source_id=None):
    return {"recording_id": recording_id,
            "source_recording_id": source_id or recording_id, "split": split}


def make_recording(root, name, base=0, original_source=None):
    folder = root / name
    folder.mkdir()
    metadata = {"recording": name}
    if original_source is not None:
        metadata["SourceRecording"] = {"Folder": f"D:\\captures\\{original_source}\\"}
    (folder / "metadata.json").write_text(json.dumps(metadata))
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

    def test_csv_split_and_training_only_normalization(self):
        rows = [assignment("train_a", "train"), assignment("train_b", "train"),
                assignment("val_a", "val"), assignment("test_a", "test")]
        sets = {split: {r["recording_id"] for r in rows if r["split"] == split} for split in SPLIT_NAMES}
        expected_train = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_split_csv(root / "dataset_splits.csv", rows)
            for i, row in enumerate(rows):
                split = row["split"]
                base = i * 10 if split == "train" else (20000 if split == "val" else -20000)
                pcm = make_recording(root, row["recording_id"], base)
                if split == "train":
                    expected_train.extend([pcm[start:start + 8] for start in (0, 16, 32)])
            dataset = build_dataset(root, sequence_length=8, max_windows_per_recording=3, split_csv=root / "dataset_splits.csv")
            for split, count in [("train", 6), ("val", 3), ("test", 3)]:
                self.assertEqual(dataset[split]["x"].shape, (count, 8, 2))
                self.assertEqual(dataset[split]["x"].dtype, np.float32)
                self.assertEqual({w["recording_id"] for w in dataset[split]["windows"]}, sets[split])
                for item in dataset[split]["windows"]:
                    self.assertEqual(item["source_recording_id"], item["recording_id"])
                    self.assertEqual(item["start_s"], item["start_frame"] / 44100)
                    self.assertEqual(item["end_s"], (item["start_frame"] + 8) / 44100)
            self.assertEqual(len(dataset["manifest"]), 4)
            self.assertEqual(dataset["ignored_recordings"], [])
            self.assertEqual(dataset["split_rows"], rows)
            self.assertEqual(dataset["split_csv"], str((root / "dataset_splits.csv").resolve()))
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

    def test_csv_changes_are_reread_in_the_same_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [assignment("a", "train"), assignment("b", "train"),
                    assignment("c", "val"), assignment("d", "test")]
            for i, row in enumerate(rows):
                make_recording(root, row["recording_id"], i * 100)
            path = root / "custom_splits.csv"
            write_split_csv(path, rows)
            first = build_dataset(root, 8, 1, split_csv=path)
            changed = [assignment("a", "val"), assignment("b", "train"),
                       assignment("c", "train"), assignment("d", "test")]
            write_split_csv(path, changed)
            second = build_dataset(root, 8, 1, split_csv=path)
            self.assertEqual({w["recording_id"] for w in first["train"]["windows"]}, {"a", "b"})
            self.assertEqual({w["recording_id"] for w in second["train"]["windows"]}, {"b", "c"})
            self.assertEqual({w["recording_id"] for w in second["val"]["windows"]}, {"a"})
            self.assertEqual(first["split_rows"], rows)
            self.assertEqual(second["split_rows"], changed)
            self.assertFalse(np.array_equal(first["train"]["x"], second["train"]["x"]))

    def test_excluded_and_unlisted_recordings_are_not_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [assignment("a", "train"), assignment("b", "val"),
                    assignment("c", "test"), assignment("excluded", "exclude")]
            write_split_csv(root / "dataset_splits.csv", rows)
            for row in rows[:3]:
                make_recording(root, row["recording_id"])
            for name in ["excluded", "unlisted"]:
                extra = root / name
                extra.mkdir()
                (extra / "metadata.json").write_text("not valid JSON")
                (extra / "audio_000000.wav").write_bytes(b"not a WAV")
            dataset = build_dataset(root, sequence_length=8, max_windows_per_recording=1, split_csv=root / "dataset_splits.csv")
            self.assertEqual(dataset["ignored_recordings"], ["excluded", "unlisted"])
            self.assertEqual(len(dataset["manifest"]), 3)
            self.assertTrue(all(len(dataset[s]["x"]) == 1 for s in SPLIT_NAMES))

    def test_missing_active_recording_is_an_explicit_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [assignment("missing_train", "train"), assignment("b", "val"), assignment("c", "test")]
            write_split_csv(root / "dataset_splits.csv", rows)
            for name in ["b", "c"]:
                make_recording(root, name)
            with self.assertRaisesRegex(ValueError, "missing recordings: missing_train"):
                build_dataset(root, split_csv=root / "dataset_splits.csv")

    def test_missing_active_metadata_or_wav_raises(self):
        for filename in ["metadata.json", "audio_000000.wav"]:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                rows = [assignment("a", "train"), assignment("b", "val"), assignment("c", "test")]
                write_split_csv(root / "dataset_splits.csv", rows)
                for row in rows:
                    make_recording(root, row["recording_id"])
                (root / "a" / filename).unlink()
                with self.assertRaises(FileNotFoundError):
                    build_dataset(root, 8, 1, split_csv=root / "dataset_splits.csv")

    def test_same_source_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [assignment("a", "train", "original"), assignment("b", "val"),
                    assignment("c", "test", "original")]
            write_split_csv(root / "dataset_splits.csv", rows)
            with self.assertRaisesRegex(ValueError, "source recording .* appears in both"):
                build_dataset(root, 8, 1, split_csv=root / "dataset_splits.csv")

    def test_metadata_source_detects_false_independent_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [assignment("a", "train"), assignment("b", "val"), assignment("c", "test")]
            write_split_csv(root / "dataset_splits.csv", rows)
            for name in ["a", "c"]:
                make_recording(root, name, original_source="same_original")
            make_recording(root, "b")
            with self.assertRaisesRegex(ValueError, "source_recording_id mismatch.*same_original"):
                build_dataset(root, 8, 1, split_csv=root / "dataset_splits.csv")

    def test_same_source_in_one_split_keeps_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [assignment("a", "train", "original"), assignment("b", "train", "original"),
                    assignment("c", "val"), assignment("d", "test")]
            write_split_csv(root / "dataset_splits.csv", rows)
            for name in ["a", "b"]:
                make_recording(root, name, original_source="original")
            for name in ["c", "d"]:
                make_recording(root, name)
            dataset = build_dataset(root, 8, 1, split_csv=root / "dataset_splits.csv")
            self.assertEqual({w["source_recording_id"] for w in dataset["train"]["windows"]}, {"original"})
            self.assertTrue(all(m["source_recording_id"] == "original" for m in dataset["manifest"] if m["split"] == "train"))


if __name__ == "__main__":
    unittest.main()
