import json

import pytest

from Integration.socket_training_bridge import WINDOW_BYTES, SocketTrainingBridge
from Integration.rtsp_cnc_metadata import build_stream_feature_vector


def test_machining_monitor_context_is_prepared_for_rtsp_training(tmp_path):
    bridge = SocketTrainingBridge(tmp_path, lambda _message: None)
    session_id = "68cdc893-6213-49bd-b15a-8dfb5961437e"
    bridge.on_start({
        "sessionId": session_id,
        "url": "rtsp://127.0.0.1:8555/audio",
        "origin": "RecordingManager",
    })

    context = {
        "Machine": "DMG MORI - DMU 50",
        "Tool": {"DiameterMm": 32.48},
        "Parameters": {
            "SpindleSpeed": {"ProgrammedRpm": 490, "AdjustmentPercent": 100, "AdjustedRpm": 490},
            "FeedRate": {"ProgrammedMmPerMin": 364, "AdjustmentPercent": 98, "AdjustedMmPerMin": 356.72},
        },
    }
    event = {"occurrence": "SpindleAdjusted", "timestamp_seconds": 4.109, "quality": 0}
    bridge.on_data({
        "sessionId": session_id,
        "sequence": 35,
        "kind": "playback",
        "data": {"Context": context, "Events": [event]},
        "positionSeconds": 4.16,
    })
    bridge.on_audio({"sessionId": session_id, "pcm16le": b"\x00" * WINDOW_BYTES})

    aligned = (tmp_path / session_id / "aligned_training.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(aligned[0])
    assert row["events"] == [event]
    assert row["context"]["context"]["context"]["Parameters"]["SpindleSpeed"]["AdjustedRpm"] == 490
    assert build_stream_feature_vector(row["context"]).tolist() == pytest.approx([490.0, 100.0, 490.0, 364.0, 98.0, 356.72, 32.48])
