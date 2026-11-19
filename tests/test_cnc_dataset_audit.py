import numpy as np
import pytest

from Dataset.Preprocessing.cnc_dataset_audit import audit_window_records


def context():
    return {
        "SchemaVersion": 6,
        "ContextId": "ctx-1",
        "MachineId": "machine-1",
        "DriverSessionId": "session-1",
        "OccurredAtUtc": "2026-11-19T10:00:00+00:00",
        "ActualSpindleRpm": 8500,
        "ActualFeedMmPerMin": 1200,
        "SPowerPercent": 42,
        "WPowerPercent": 18,
        "ToolDiameterMm": 12,
        "ToolTeethCount": 4,
        "ToolRemainingUsePercent": 78,
    }


def record(**overrides):
    value = {
        "sample_id": "sample-1",
        "recording_id": "recording-1",
        "source_recording_id": "source-1",
        "split": "train",
        "sample_rate": 44100,
        "start_frame": 0,
        "end_frame_exclusive": 220500,
        "label": "good",
        "label_index": 0,
        "pcm": np.zeros((220500, 2), dtype=np.float32),
        "context": context(),
    }
    value.update(overrides)
    return value


def test_valid_service_record_is_prepared_with_raw_audio_and_cnc_vector():
    result = audit_window_records([record()], ["good", "bad"])

    assert result["report"]["ready_for_training"] is True
    assert result["report"]["counts"] == {
        "received": 1,
        "accepted": 1,
        "rejected": 0,
        "labels": {"good": 1},
        "splits": {"train": 1},
    }
    prepared = result["accepted_records"][0]
    assert prepared["pcm"].shape == (220500, 2)
    assert prepared["cnc_features"].shape == (7,)
    assert prepared["label_index"] == 0


def test_pcm16_bytes_are_supported_at_service_boundary():
    pcm = np.zeros((220500, 2), dtype="<i2")
    result = audit_window_records([record(pcm=None, pcm16_le=pcm.tobytes())], ["good", "bad"])

    assert result["report"]["counts"]["accepted"] == 1


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("sample_rate", 48000, "sample_rate_mismatch"),
        ("end_frame_exclusive", 220499, "invalid_window"),
        ("label_index", 1, "label_mismatch"),
        ("context", {"bad": []}, "invalid_context"),
    ],
)
def test_invalid_records_are_rejected_with_reason(field, value, code):
    result = audit_window_records([record(**{field: value})], ["good", "bad"])

    assert result["report"]["ready_for_training"] is False
    codes = {item["code"] for item in result["report"]["rejected_records"][0]["errors"]}
    assert code in codes


def test_source_recording_cannot_cross_splits():
    result = audit_window_records(
        [record(sample_id="train-1"), record(sample_id="val-1", split="val")],
        ["good", "bad"],
    )

    assert result["report"]["counts"]["accepted"] == 0
    assert all(
        any(error["code"] == "source_split_leakage" for error in item["errors"])
        for item in result["report"]["rejected_records"]
    )


def test_service_record_requires_one_complete_context_and_label():
    value = record()
    value["context"] = dict(context())
    value["context"].pop("ActualFeedMmPerMin")

    result = audit_window_records([value], ["good", "bad"])

    assert result["report"]["counts"]["rejected"] == 1
    assert result["report"]["rejected_records"][0]["errors"][0]["code"] == "invalid_context"
