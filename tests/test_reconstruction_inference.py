from reconstruction_inference import _incidents, _state_for_score


def _baseline():
    return {
        "minimum_global_samples": 20,
        "minimum_context_samples": 3,
        "global": {"count": 30, "p95": 5.0, "p99": 8.0},
        "by_context": {"operation": {"Turning": {"count": 10, "p95": 4.0, "p99": 6.0}}},
        "validation_calibration": {"calibrated_threshold": None},
    }


def test_state_uses_operation_baseline_and_statistical_confidence():
    state, threshold, source, confidence = _state_for_score(7.0, _baseline(), {"operation": "Turning"})
    assert state == "anomalous"
    assert threshold == 6.0
    assert source == "operation"
    assert 0.0 <= confidence <= 1.0


def test_incidents_group_consecutive_anomalous_windows_and_keep_context():
    rows = [
        {"start_s": 0.0, "end_s": 1.0, "score": 9.0, "state": "anomalous",
         "context": {"operation": "Turning", "tool": "T1", "parameters": {"rpm": 1000}}, "events": []},
        {"start_s": 1.0, "end_s": 2.0, "score": 10.0, "state": "out_of_distribution",
         "context": {"operation": "Turning", "tool": "T1", "parameters": {}}, "events": [{"label": "review"}]},
        {"start_s": 2.0, "end_s": 3.0, "score": 1.0, "state": "normal",
         "context": {"operation": "Turning", "tool": "T1", "parameters": {}}, "events": []},
    ]
    incidents = _incidents(rows)
    assert len(incidents) == 1
    assert incidents[0]["duration_s"] == 2.0
    assert incidents[0]["window_count"] == 2
    assert incidents[0]["operations"] == ["Turning"]
    assert incidents[0]["events"][0]["label"] == "review"
