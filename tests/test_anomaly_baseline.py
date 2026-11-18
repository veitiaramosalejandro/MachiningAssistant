import json

from anomaly_baseline import build_baseline_report, score_statistics


def _rows():
    rows = []
    for index in range(20):
        rows.append({
            "recording_id": "rec-1", "split": "train", "start_s": index,
            "end_s": index + 1, "best_mse_standardized": float(index + 1),
        })
    for index, score in enumerate((2.0, 3.0, 4.0, 30.0)):
        rows.append({
            "recording_id": "rec-1", "split": "val", "start_s": 30 + index,
            "end_s": 31 + index, "best_mse_standardized": score,
            "is_anomaly": score > 10,
        })
    rows.append({
        "recording_id": "rec-1", "split": "test", "start_s": 40,
        "end_s": 41, "best_mse_standardized": 50.0,
    })
    return rows


def test_score_statistics_returns_required_percentiles():
    stats = score_statistics([1, 2, 3, 4, 5])
    assert stats["count"] == 5
    assert stats["mean"] == 3.0
    assert stats["p95"] > stats["p90"]
    assert stats["p99"] <= stats["max"]


def test_baseline_report_uses_train_scores_and_explicit_validation_labels(tmp_path):
    folder = tmp_path / "rec-1"
    folder.mkdir()
    (folder / "metadata.json").write_text(json.dumps({"MachineName": "M-01"}), encoding="utf-8")
    (folder / "contexts.json").write_text(json.dumps({"contexts": [{
        "timestamp_seconds": 0, "context": {
            "MachineName": "M-01", "Operation": "Turning", "Material": "Steel",
            "ProgramName": "P-01", "MachineState": "Cutting", "Tool": {"name": "T1"},
        },
    }]}), encoding="utf-8")
    report = build_baseline_report(_rows(), tmp_path, min_context_samples=3)
    assert report["status"] == "ready"
    assert report["global"]["count"] == 20
    assert 19.0 < report["global"]["p99"] <= 20.0
    assert report["by_context"]["operation"]["Turning"]["count"] == 20
    assert report["validation_calibration"]["status"] == "label_calibrated"
    assert report["validation_calibration"]["metrics"]["true_positive"] == 1
    assert {row["state"] for row in report["window_states"]}
