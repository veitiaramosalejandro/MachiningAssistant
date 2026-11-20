"""Build the RTSP multimodal synthetic six-class dataset.

Each complete captured window is kept as the base signal and produces six
labels using the same controlled right-channel sine injection convention as
the initial synthetic dataset. CNC metadata is copied unchanged to every
variant. ``event.quality`` is never read as a target.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
import shutil
import wave

import numpy as np

from Integration.rtsp_cnc_metadata import build_stream_feature_vector


SAMPLE_RATE = 44100
CHANNELS = 2
VALID_FRAMES = 220500
# RTSP PCM is interleaved as [sound, vibration] for every frame. The model
# receives both channels; synthetic excitation is applied only to vibration.
SOUND_CHANNEL = 0
VIBRATION_CHANNEL = 1
INJECTION_CHANNEL = VIBRATION_CHANNEL
SINE_FREQUENCY_HZ = 1000.0
# Synthetic engineering tiers in SYNTHETIC_CLASS_NAMES order. These are not
# calibrated machining-quality probabilities or real fault measurements.
AMPLITUDE_RATIOS = (0.0, 0.5, 1.0, 1.5, 2.25, 3.0)
AMPLITUDE_JITTER_FRACTION = 0.1
SEED = 20261109
SYNTHETIC_CLASS_NAMES =  ('excellent', 'good', 'sufficient', 'bad', 'very_bad', 'broken')


def _read_json_lines(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fields = [
        "sample_id", "base_sample_id", "recording_id", "source_recording_id",
        "split", "label", "label_index", "wav_file", "frame_count",
        "start_frame", "end_frame_exclusive", "start_seconds",
        "end_seconds_exclusive", "duration_seconds",
    ]
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as reader:
        expected = (SAMPLE_RATE, CHANNELS, 2, "NONE", VALID_FRAMES)
        actual = (reader.getframerate(), reader.getnchannels(), reader.getsampwidth(), reader.getcomptype(), reader.getnframes())
        if actual != expected:
            raise ValueError(f"{path}: expected 44100 Hz stereo PCM16 with 220500 frames")
        payload = reader.readframes(VALID_FRAMES)
    return np.frombuffer(payload, dtype="<i2").reshape(VALID_FRAMES, CHANNELS).copy()


def _write_wav(path: Path, pcm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(CHANNELS)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(np.asarray(pcm, dtype="<i2").tobytes())


def _reserve_pcm16_headroom(pcm: np.ndarray) -> tuple[np.ndarray, bool]:
    """Attenuate already saturated capture audio before synthetic injection."""
    peak = float(np.max(np.abs(pcm.astype(np.float64))))
    if peak <= 32700.0:
        return pcm, False
    scale = min(1.0, (32700.0 / peak))
    if scale >= 1.0:
        return pcm, False
    return np.rint(pcm.astype(np.float64) * scale).astype("<i2"), True


def _random_parameters(sample_id: str) -> tuple[float, float]:
    identity = sample_id.encode("utf-8")
    rng = np.random.default_rng(np.random.SeedSequence([SEED, *identity]))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    jitter = float(rng.uniform(1.0 - AMPLITUDE_JITTER_FRACTION, 1.0 + AMPLITUDE_JITTER_FRACTION))
    return phase, jitter


def _make_variant(original: np.ndarray, label: str, base_id: str, reference_rms: float) -> tuple[np.ndarray, dict]:
    label_index = SYNTHETIC_CLASS_NAMES.index(label)
    phase, jitter = _random_parameters(base_id)
    ratio = AMPLITUDE_RATIOS[label_index]
    if ratio == 0:
        return original.copy(), {
            "generation_frequency_hz": SINE_FREQUENCY_HZ,
            "injection_channel": INJECTION_CHANNEL,
            "peak_amplitude_ratio": ratio,
            "peak_amplitude_pcm": 0.0,
            "phase_radians": phase,
            "jitter_multiplier": jitter,
            "vibration_changed_frame_count": 0,
        }
    time_axis = np.arange(VALID_FRAMES, dtype=np.float64) / SAMPLE_RATE
    sine = np.sin(2.0 * np.pi * SINE_FREQUENCY_HZ * time_axis + phase)
    amplitude = ratio * reference_rms * jitter
    result = original.copy()
    values = np.rint(original[:, INJECTION_CHANNEL].astype(np.float64) + amplitude * sine)
    if values.min() < -32768 or values.max() > 32767:
        raise ValueError(f"{base_id} / {label}: synthetic PCM16 injection overflows")
    result[:, INJECTION_CHANNEL] = values.astype("<i2")
    return result, {
        "generation_frequency_hz": SINE_FREQUENCY_HZ,
        "injection_channel": INJECTION_CHANNEL,
        "peak_amplitude_ratio": ratio,
        "peak_amplitude_pcm": amplitude,
        "phase_radians": phase,
        "jitter_multiplier": jitter,
        "vibration_changed_frame_count": int(np.count_nonzero(result[:, INJECTION_CHANNEL] != original[:, INJECTION_CHANNEL])),
    }


def build_rtsp_dataset(*, session_dir: Path, aligned_path: Path, output_dir: Path) -> dict:
    """Create synthetic four-class audio plus the independent stream context."""
    session_dir = session_dir.expanduser().resolve()
    aligned_path = aligned_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    rows = _read_json_lines(aligned_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    bases = []
    rejected = []
    for index, row in enumerate(rows):
        audio_wav_file = row.get("audio_wav_file")
        source = Path(audio_wav_file).expanduser() if isinstance(audio_wav_file, str) else None
        if source is not None and not source.is_absolute():
            source = (session_dir / source).resolve()
        context = row.get("context")
        if source is None or not source.is_file():
            rejected.append({"index": index, "reason": "audio_window_missing"})
            continue
        if not isinstance(context, dict):
            rejected.append({"index": index, "reason": "stream_metadata_missing"})
            continue
        try:
            build_stream_feature_vector(context)
            pcm = _read_wav(source)
            pcm, attenuated = _reserve_pcm16_headroom(pcm)
        except (OSError, TypeError, ValueError) as exc:
            rejected.append({"index": index, "reason": "stream_input_invalid", "error": str(exc)})
            continue
        bases.append({"index": index, "row": row, "source": source, "context": context, "pcm": pcm, "attenuated": attenuated})

    train_bases = [item for position, item in enumerate(bases) if position % 3 == 0]
    if not train_bases:
        report = {"ready": False, "status": "blocked", "reason": "No valid RTSP windows available for training reference RMS", "accepted_base_windows": 0, "rejected_samples": len(rejected)}
        _write_json(output_dir / "dataset_report.json", report)
        return report
    original_reference_rms = float(np.median([
        np.sqrt(np.mean(item["pcm"][:, 1].astype(np.float64) ** 2)) for item in train_bases
    ]))
    if not np.isfinite(original_reference_rms) or original_reference_rms <= 0:
        raise ValueError("Synthetic RTSP reference RMS must be positive")
    # Keep every generated PCM16 sample valid without clipping. This safety
    # limit is based on signal headroom, not on labels or model predictions.
    headrooms = []
    for base in bases:
        right = base["pcm"][:, INJECTION_CHANNEL].astype(np.float64)
        headrooms.append(min(32767.0 - float(right.max()), 32768.0 + float(right.min())))
    # Leave one PCM count of margin for rounding at the extrema.
    available_peak = min(headrooms) - 1.0
    maximum_requested_peak = max(AMPLITUDE_RATIOS) * (1.0 + AMPLITUDE_JITTER_FRACTION)
    if available_peak <= 0 or maximum_requested_peak <= 0:
        raise ValueError("RTSP audio has no PCM16 headroom for synthetic variants")
    reference_rms = min(original_reference_rms, available_peak / maximum_requested_peak)
    if reference_rms <= 0 or not np.isfinite(reference_rms):
        raise ValueError("Synthetic RTSP reference RMS is unsafe for PCM16 generation")

    samples = []
    for position, base in enumerate(bases):
        split = ("train", "val", "test")[position % 3]
        base_id = f"stream_window_{base['index']:06d}"
        for label_index, label in enumerate(SYNTHETIC_CLASS_NAMES):
            sample_id = f"{base_id}_{label}"
            variant, generation = _make_variant(base["pcm"], label, base_id, reference_rms)
            target = audio_dir / f"{sample_id}.wav"
            _write_wav(target, variant)
            start_seconds = float(base["row"].get("window_start_seconds") or 0.0)
            end_seconds = start_seconds + VALID_FRAMES / SAMPLE_RATE
            start_frame = int(round(start_seconds * SAMPLE_RATE))
            recording_id = (
                base["context"].get("recording_id")
                or base["context"].get("session", {}).get("recording_id")
                or base_id
            )
            samples.append({
                "sample_id": sample_id,
                "base_sample_id": base_id,
                "recording_id": str(recording_id),
                "source_recording_id": str(recording_id),
                "split": split,
                "label": label,
                "label_index": label_index,
                "wav_file": str(Path("audio") / target.name),
                "frame_count": VALID_FRAMES,
                "start_frame": start_frame,
                "end_frame_exclusive": start_frame + VALID_FRAMES,
                "start_seconds": start_frame / SAMPLE_RATE,
                "end_seconds_exclusive": (start_frame + VALID_FRAMES) / SAMPLE_RATE,
                "duration_seconds": VALID_FRAMES / SAMPLE_RATE,
                "context": base["context"],
                "synthetic_only": True,
                "label_semantics": "synthetic_sine_amplitude_tier",
                **generation,
            })

    sample_fields = (
        "sample_id", "base_sample_id", "recording_id", "source_recording_id",
        "split", "label", "label_index", "wav_file", "frame_count",
        "start_frame", "end_frame_exclusive", "start_seconds",
        "end_seconds_exclusive", "duration_seconds",
    )
    sample_rows = [{key: item[key] for key in sample_fields} for item in samples]
    context_records = [{"sample_id": item["sample_id"], "context": item["context"]} for item in samples]
    _write_json(output_dir / "labels.json", {str(index): label for index, label in enumerate(SYNTHETIC_CLASS_NAMES)})
    _write_csv(output_dir / "samples.csv", sample_rows)
    _write_json(output_dir / "contexts.json", context_records)
    split_counts = Counter(item["split"] for item in samples)
    class_counts = Counter(item["label"] for item in samples)
    report = {
        "ready": len(bases) >= 3 and all(split_counts[split] > 0 for split in ("train", "val", "test")),
        "dataset_dir": str(output_dir),
        "samples_csv": str(output_dir / "samples.csv"),
        "contexts_json": str(output_dir / "contexts.json"),
        "labels_json": str(output_dir / "labels.json"),
        "accepted_base_windows": len(bases),
        "generated_samples": len(samples),
        "rejected_samples": len(rejected),
        "class_counts": dict(class_counts),
        "split_counts": dict(split_counts),
        "required_classes": list(SYNTHETIC_CLASS_NAMES),
        "context_contract": "rtsp_stream",
        "label_source": "synthetic_audio_transform",
        "synthetic_only": True,
        "event_quality_used_as_target": False,
        "reference_rms_pcm": reference_rms,
        "original_reference_rms_pcm": original_reference_rms,
        "pcm16_headroom_limited": reference_rms < original_reference_rms,
        "minimum_right_channel_headroom_pcm": available_peak,
        "attenuated_base_windows": sum(1 for item in bases if item["attenuated"]),
        "rejections": rejected[-100:],
    }
    _write_json(output_dir / "dataset_report.json", report)
    return report


__all__ = ["build_rtsp_dataset"]
