"""Read the current recording split CSV without changing source signals.

Windows are split by recording before normalization. Both channels retain their
PCM counts; there is no filtering, resampling, or per-window normalization.
"""
from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from Dataset.Preprocessing.dataset_splits import SPLIT_NAMES, read_split_csv


DEFAULT_SOURCE = PROJECT_ROOT / "Dataset" / "Data"
DEFAULT_SPLIT_CSV = Path(__file__).resolve().parent / "dataset_splits.csv"
SAMPLE_RATE = 44100


def read_wav(path: str | Path) -> tuple[int, np.ndarray]:
    """Return 44.1 kHz stereo PCM16 as float32 [frames, 2] PCM counts."""
    path = Path(path)
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.getnframes()
        if (
            sample_rate != SAMPLE_RATE
            or channels != 2
            or sample_width != 2
            or wav.getcomptype() != "NONE"
        ):
            raise ValueError(
                f"{path}: expected 44100 Hz, uncompressed PCM16, 2 channels; "
                f"got {sample_rate} Hz, {sample_width * 8} bits, {channels} channels"
            )
        payload = wav.readframes(frames)
    if len(payload) != frames * channels * sample_width:
        raise ValueError(f"{path}: WAV payload is shorter than its declared frame count")
    pcm = np.frombuffer(payload, dtype="<i2").reshape(frames, 2).astype(np.float32)
    return sample_rate, pcm


def select_window_starts(
    frame_count: int,
    sequence_length: int = 2048,
    max_windows_per_recording: int = 16,
) -> np.ndarray:
    """Select evenly spaced complete windows from a nonoverlapping grid.

    A positive cap is mandatory: zero does not mean "load every window".
    The final incomplete window is excluded.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer")
    if max_windows_per_recording < 1:
        raise ValueError("max_windows_per_recording must be a positive integer")
    window_count = frame_count // sequence_length
    if window_count < 1:
        raise ValueError(
            f"Recording has {frame_count} frames, fewer than sequence_length={sequence_length}"
        )
    selected_count = min(window_count, max_windows_per_recording)
    indices = np.linspace(0, window_count - 1, num=selected_count, dtype=np.int64)
    return indices * sequence_length


def build_dataset(
    source: str | Path = DEFAULT_SOURCE,
    sequence_length: int = 2048,
    max_windows_per_recording: int = 16,
    split_csv: str | Path | None = None,
) -> dict:
    """Reread the split CSV and load a capped set of windows per selected recording.

    Each split contains ``x`` (float32 [N, T, 2]) and matching ``windows``.
    ``manifest`` records the original metadata, actual WAV length, and split.
    Other recording directories are returned in ``ignored_recordings`` and
    are not opened. The returned ``split_rows`` are the assignments used by
    this invocation, including exclusions, for saving a reproducible snapshot.
    """
    source = Path(source).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {source}")
    if sequence_length < 1 or max_windows_per_recording < 1:
        raise ValueError("sequence_length and max_windows_per_recording must be positive")

    split_path = (Path(split_csv) if split_csv is not None else DEFAULT_SPLIT_CSV).resolve()
    split_rows = read_split_csv(split_path)
    active_rows = [row for row in split_rows if row["split"] in SPLIT_NAMES]
    found = {p.name for p in source.iterdir() if p.is_dir() and not p.name.startswith(".")}
    expected = {row["recording_id"] for row in active_rows}
    missing = sorted(expected - found)
    ignored = sorted(found - expected)
    if missing:
        raise ValueError(
            "missing recordings: " + ", ".join(missing)
            + ". Every active recording in the split CSV is required."
        )

    dataset = {split: {"x": [], "windows": []} for split in SPLIT_NAMES}
    manifest = []
    for row in active_rows:
        recording_id = row["recording_id"]
        directory = source / recording_id
        metadata_path = directory / "metadata.json"
        with metadata_path.open(encoding="utf-8-sig") as handle:
            metadata = json.load(handle)
        original = metadata.get("SourceRecording")
        if original is None:
            actual_source_id = recording_id
        else:
            folder = original.get("Folder") if isinstance(original, dict) else None
            if not isinstance(folder, str) or not folder.strip().strip("/\\"):
                raise ValueError(f"{metadata_path}: SourceRecording.Folder must identify the original recording")
            actual_source_id = folder.strip().replace("\\", "/").rstrip("/").split("/")[-1]
        if row["source_recording_id"] != actual_source_id:
            raise ValueError(
                f"source_recording_id mismatch for {recording_id}: "
                f"CSV declares {row['source_recording_id']!r}, metadata identifies {actual_source_id!r}"
            )
        sample_rate, pcm = read_wav(directory / "audio_000000.wav")
        starts = select_window_starts(len(pcm), sequence_length, max_windows_per_recording)
        split = row["split"]
        for value in starts:
            start = int(value)
            end = start + sequence_length
            # Copy the small selected window, so it cannot retain the full WAV.
            dataset[split]["x"].append(pcm[start:end].copy())
            dataset[split]["windows"].append({
                "recording_id": recording_id,
                "source_recording_id": actual_source_id,
                "start_frame": start,
                "start_s": start / sample_rate,
                "end_s": end / sample_rate,
            })
        manifest.append({
            "recording_id": recording_id,
            "source_recording_id": actual_source_id,
            "split": split,
            "metadata": metadata,
            "sample_rate": sample_rate,
            "frame_count": len(pcm),
            "duration_s": len(pcm) / sample_rate,
            "window_count": len(starts),
        })
    for split in SPLIT_NAMES:
        dataset[split]["x"] = np.stack(dataset[split]["x"]).astype(np.float32, copy=False)
    dataset["manifest"] = manifest
    dataset["ignored_recordings"] = ignored
    dataset["split_csv"] = str(split_path)
    dataset["split_rows"] = split_rows
    return dataset


class ChannelNormalizer:
    """One affine transform per channel, fitted only to supplied training data."""

    def __init__(self, mean_pcm, scale_pcm):
        self.mean_pcm = np.asarray(mean_pcm, dtype=np.float64)
        self.scale_pcm = np.asarray(scale_pcm, dtype=np.float64)
        if self.mean_pcm.shape != (2,) or self.scale_pcm.shape != (2,):
            raise ValueError("mean_pcm and scale_pcm must each contain two channel values")
        if not np.all(np.isfinite(self.mean_pcm)) or not np.all(np.isfinite(self.scale_pcm)):
            raise ValueError("Normalization parameters must be finite")
        if np.any(self.scale_pcm <= 0):
            raise ValueError("Channel scales must be positive")

    @classmethod
    def fit(cls, train_x: np.ndarray) -> "ChannelNormalizer":
        """Fit population mean/std over N and T using float64 arithmetic."""
        values = np.asarray(train_x, dtype=np.float64)
        if values.ndim != 3 or values.shape[-1] != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("train_x must have nonempty shape [N, T, 2]")
        if not np.all(np.isfinite(values)):
            raise ValueError("train_x must contain only finite PCM values")
        mean = values.mean(axis=(0, 1), dtype=np.float64)
        scale = np.maximum(values.std(axis=(0, 1), dtype=np.float64), 1.0)
        return cls(mean, scale)

    @classmethod
    def full_scale(cls) -> "ChannelNormalizer":
        """Use the fixed PCM16 scale without estimating dataset statistics."""
        return cls([0.0, 0.0], [32768.0, 32768.0])

    def transform(self, x: np.ndarray) -> np.ndarray:
        values = self._values(x)
        return ((values - self.mean_pcm) / self.scale_pcm).astype(np.float32)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        values = self._values(x)
        return (values * self.scale_pcm + self.mean_pcm).astype(np.float32)

    @staticmethod
    def _values(x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        if values.ndim < 2 or values.shape[-1] != 2:
            raise ValueError("Expected an array with two channels in its last dimension")
        return values

    def to_dict(self) -> dict:
        return {"mean_pcm": self.mean_pcm.tolist(), "scale_pcm": self.scale_pcm.tolist()}

    @classmethod
    def from_dict(cls, values: dict) -> "ChannelNormalizer":
        return cls(values["mean_pcm"], values["scale_pcm"])
