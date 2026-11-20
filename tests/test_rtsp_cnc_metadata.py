import pytest

from Integration.rtsp_cnc_metadata import (
    STREAM_FEATURE_NAMES,
    StreamCncMetadata,
    StreamFeatureNormalizer,
    build_stream_feature_vector,
)


def observed_snapshot():
    return {
        "type": "snapshot",
        "reason": "start",
        "recordingId": "20261021_122049",
        "timestampSeconds": 0,
        "metadata": {
            "StartTime": "2026-10-21T12:20:49.803+01:00",
            "Mode": "VIDEO_AUDIO",
            "AudioFile": "audio_%06d.wav",
            "AudioFiles": ["audio_000000.wav"],
            "Context": {"Machine": "DMG MORI - DMU 50"},
        },
        "context": {
            "id": "initial",
            "timestamp_seconds": 0,
            "context": {
                "Machine": "DMG MORI - DMU 50",
                "Operator": "Operador 1",
                "Work": "2411_G801A_Desb1_2",
                "Part": "G802A",
                "PartMaterial": "Aço",
                "Operation": "Desbaste",
                "Tool": {
                    "Tool": "PAULBIT-D25-R1.8-L60",
                    "Type": "Fresa desbaste",
                    "DiameterMm": 25,
                    "Flutes": None,
                },
                "Parameters": {
                    "SpindleSpeed": {"ProgrammedRpm": 2546, "AdjustmentPercent": 100, "AdjustedRpm": 2546},
                    "FeedRate": {"ProgrammedMmPerMin": 9170, "AdjustmentPercent": 54.11, "AdjustedMmPerMin": 4962},
                    "StkXY": None, "StkZ": None, "IncXY": None, "IncZ": None,
                },
            },
        },
        "parameters": {
            "SpindleSpeed": {"ProgrammedRpm": 2546, "AdjustmentPercent": 100, "AdjustedRpm": 2546},
            "FeedRate": {"ProgrammedMmPerMin": 9170, "AdjustmentPercent": 54.11, "AdjustedMmPerMin": 4962},
            "StkXY": None, "StkZ": None, "IncXY": None, "IncZ": None,
        },
        "quality": 0,
    }


def test_complete_snapshot_payload_is_represented_without_machining_context_mapping():
    payload = StreamCncMetadata.from_message(observed_snapshot())

    assert payload.type == "snapshot"
    assert payload.recordingId == "20261021_122049"
    assert payload.metadata.Mode == "VIDEO_AUDIO"
    assert payload.context.context.Machine == "DMG MORI - DMU 50"
    assert payload.context.context.Parameters.FeedRate.AdjustedMmPerMin == 4962
    assert payload.parameters.SpindleSpeed.ProgrammedRpm == 2546
    assert payload.context.context.Tool.DiameterMm == 25
    assert payload.context.context.Tool.Flutes is None
    assert payload.quality == 0


def test_event_payload_is_represented_and_quality_remains_annotation():
    payload = StreamCncMetadata.from_message({
        "type": "event",
        "recordingId": "20261021_122049",
        "timestampSeconds": 33.16,
        "sequence": 35,
        "event": {
            "id": "event-1",
            "timestamp_seconds": 33.16,
            "quality": 0,
            "occurrence": "SpindleAdjusted",
            "value": 100,
            "machine_activity": "idle",
            "machine_state": "idle",
            "parameters": {"SpindleSpeed": {"AdjustedRpm": 2546}},
            "parameter_changes": {"SpindleSpeed": []},
            "event_changes": {"quality": False},
        },
    })

    assert payload.event.occurrence == "SpindleAdjusted"
    assert payload.event.value == 100
    assert payload.event.quality == 0
    assert payload.event.parameters.SpindleSpeed.AdjustedRpm == 2546


def test_round_trip_keeps_stream_contract_fields_and_does_not_call_machining_context():
    payload = StreamCncMetadata.from_message(observed_snapshot())
    mapped = payload.to_mapping()

    assert mapped["context"]["context"]["Tool"]["DiameterMm"] == 25
    assert mapped["parameters"]["StkXY"] is None
    assert mapped["quality"] == 0


def test_stream_model_vector_uses_received_cnc_fields_not_quality():
    payload = observed_snapshot()
    first = build_stream_feature_vector(payload)
    payload["quality"] = 5
    second = build_stream_feature_vector(payload)

    assert list(STREAM_FEATURE_NAMES) == [
        "ProgrammedRpm", "SpindleAdjustmentPercent", "AdjustedRpm",
        "ProgrammedMmPerMin", "FeedAdjustmentPercent", "AdjustedMmPerMin",
        "ToolDiameterMm",
    ]
    assert first.tolist() == pytest.approx(second.tolist())
    assert first.tolist() == pytest.approx([2546.0, 100.0, 2546.0, 9170.0, 54.11, 4962.0, 25.0])


def test_stream_normalizer_serializes_independent_contract():
    vector = build_stream_feature_vector(observed_snapshot())
    normalizer = StreamFeatureNormalizer.fit([vector, vector + 1])
    restored = StreamFeatureNormalizer.from_dict(normalizer.to_dict())

    assert restored.feature_names == STREAM_FEATURE_NAMES
    assert restored.transform(vector).shape == (len(STREAM_FEATURE_NAMES),)
