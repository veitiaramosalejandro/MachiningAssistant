import json
import wave

import numpy as np

from Integration.rtsp_training_dataset import build_rtsp_dataset


def write_stream_wav(path, right_value=1000):
    pcm = np.zeros((220500, 2), dtype="<i2")
    pcm[:, 0] = 100
    pcm[:, 1] = right_value
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(44100)
        writer.writeframes(pcm.tobytes())
    return pcm


def stream_context():
    return {
        "type": "snapshot",
        "recordingId": "stream-1",
        "timestampSeconds": 0,
        "context": {
            "id": "ctx-1",
            "timestamp_seconds": 0,
            "context": {
                "Machine": "test-machine",
                "Tool": {"Tool": "D25", "DiameterMm": 25},
                "Parameters": {
                    "SpindleSpeed": {"ProgrammedRpm": 2546, "AdjustmentPercent": 100, "AdjustedRpm": 2546},
                    "FeedRate": {"ProgrammedMmPerMin": 9170, "AdjustmentPercent": 54.11, "AdjustedMmPerMin": 4962},
                },
            },
        },
        "parameters": {
            "SpindleSpeed": {"ProgrammedRpm": 2546, "AdjustmentPercent": 100, "AdjustedRpm": 2546},
            "FeedRate": {"ProgrammedMmPerMin": 9170, "AdjustmentPercent": 54.11, "AdjustedMmPerMin": 4962},
        },
        "quality": 0,
    }


def test_rtsp_dataset_generates_four_synthetic_labels_and_keeps_context(tmp_path):
    rows = []
    for index in range(3):
        wav = tmp_path / f"window_{index}.wav"
        write_stream_wav(wav)
        rows.append({
            "window_start_seconds": index * 5,
            "window_end_seconds": (index + 1) * 5,
            "audio_wav_file": str(wav),
            "context": stream_context(),
        })
    aligned = tmp_path / "aligned_training.jsonl"
    aligned.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = build_rtsp_dataset(
        session_dir=tmp_path,
        aligned_path=aligned,
        output_dir=tmp_path / "training_dataset",
    )

    assert report["ready"] is True
    assert report["synthetic_only"] is True
    assert report["event_quality_used_as_target"] is False
    assert report["generated_samples"] == 18
    assert report["class_counts"] == {
        "excellent": 3, "good": 3, "sufficient": 3,
        "bad": 3, "very_bad": 3, "broken": 3,
    }
    assert report["split_counts"] == {"train": 6, "val": 6, "test": 6}
    contexts = json.loads((tmp_path / "training_dataset" / "contexts.json").read_text(encoding="utf-8"))
    assert all(record["context"]["quality"] == 0 for record in contexts)
    assert all(record["context"]["type"] == "snapshot" for record in contexts)


def test_rtsp_dataset_reduces_synthetic_amplitude_to_avoid_pcm16_overflow(tmp_path):
    rows = []
    for index in range(3):
        wav = tmp_path / f"high_{index}.wav"
        write_stream_wav(wav, right_value=32760)
        rows.append({"audio_wav_file": str(wav), "context": stream_context()})
    aligned = tmp_path / "aligned_training.jsonl"
    aligned.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = build_rtsp_dataset(session_dir=tmp_path, aligned_path=aligned, output_dir=tmp_path / "training_dataset")

    assert report["ready"] is True
    assert report["pcm16_headroom_limited"] is True
    assert report["reference_rms_pcm"] < report["original_reference_rms_pcm"]
