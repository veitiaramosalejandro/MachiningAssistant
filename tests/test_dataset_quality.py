import json

import numpy as np

from Dataset.Preprocessing.dataset_quality import validate_recording


def _write_documents(root, *, timestamp=0.0, rpm=1000):
    (root / "metadata.json").write_text(json.dumps({
        "ExecutionId": "rec-1",
        "DurationSeconds": 8 / 44100,
        "StartTime": "2026-11-18T10:00:00+00:00",
        "EndTime": "2026-11-18T10:00:00.001+00:00",
    }), encoding="utf-8")
    (root / "contexts.json").write_text(json.dumps({
        "schema_version": 1,
        "time_reference": "playback_seconds",
        "contexts": [{"id": "ctx-1", "timestamp_seconds": timestamp,
                       "context": {"Parameters": {"SpindleSpeed": {"ProgrammedRpm": rpm}}}}],
    }), encoding="utf-8")
    (root / "events.json").write_text(json.dumps({
        "schema_version": 1,
        "time_reference": "playback_seconds",
        "annotations": [{"id": "event-1", "timestamp_seconds": timestamp,
                          "driver": {"ContextId": "ctx-1"}}],
    }), encoding="utf-8")
    (root / "playback.json").write_text("{}", encoding="utf-8")


def test_valid_recording_and_context_documents_are_accepted(tmp_path):
    _write_documents(tmp_path)
    result = validate_recording(
        tmp_path, "rec-1", "rec-1", "train",
        metadata=json.loads((tmp_path / "metadata.json").read_text()),
        pcm=np.zeros((8, 2), dtype=np.float32), sample_rate=44100,
    )
    assert result["status"] == "valid"
    assert result["errors"] == []


def test_out_of_range_context_and_invalid_cnc_value_are_reported(tmp_path):
    _write_documents(tmp_path, timestamp=2.0, rpm=-1)
    result = validate_recording(
        tmp_path, "rec-1", "rec-1", "train",
        metadata=json.loads((tmp_path / "metadata.json").read_text()),
        pcm=np.zeros((8, 2), dtype=np.float32), sample_rate=44100,
    )
    codes = {issue["code"] for issue in result["errors"]}
    assert "timestamp_out_of_range" in codes
    assert "negative_cnc_value" in codes


def test_missing_optional_documents_are_warnings(tmp_path):
    metadata = {"ExecutionId": "rec-1", "DurationSeconds": 8 / 44100}
    result = validate_recording(
        tmp_path, "rec-1", "rec-1", "train", metadata=metadata,
        pcm=np.zeros((8, 2), dtype=np.float32), sample_rate=44100,
    )
    assert result["status"] == "warning"
    assert result["errors"] == []
    assert {issue["code"] for issue in result["warnings"]} == {"missing_json"}
