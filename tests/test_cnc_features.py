import numpy as np
import pytest

from Dataset.Preprocessing.cnc_features import (
    FEATURE_NAMES,
    SCALAR_CONTEXT_FIELDS,
    build_feature_vector,
    feature_schema,
    validate_context,
)


def complete_context():
    return {
        "SchemaVersion": 6,
        "Simulation": 0,
        "ContextId": "ctx-1",
        "MachineId": "machine-1",
        "DriverSessionId": "session-1",
        "Sequence": 12,
        "OccurredAtUtc": "2026-11-19T10:00:00+00:00",
        "ExecutionId": "execution-1",
        "ProgramName": "P-01",
        "MachineName": "M-01",
        "Operation": "roughing",
        "Strategy": "adaptive",
        "PartMaterial": "Aluminium",
        "Tool": "T01",
        "ToolType": "EndMill",
        "ToolMaterial": "CementedCarbide",
        "ActualSpindleRpm": 8500,
        "ActualFeedMmPerMin": 1200,
        "SPowerPercent": 42,
        "WPowerPercent": 18,
        "ToolDiameterMm": 12.0,
        "ToolTeethCount": 4,
        "ToolRemainingUsePercent": 78,
    }


def test_build_feature_vector_uses_stable_order_and_float32():
    vector = build_feature_vector(complete_context())

    assert vector.dtype == np.float32
    assert vector.shape == (len(FEATURE_NAMES),)
    assert vector.flags.c_contiguous
    assert vector.tolist() == [8500.0, 1200.0, 42.0, 18.0, 12.0, 4.0, 78.0]


def test_validate_context_preserves_all_received_scalar_fields():
    values = validate_context(complete_context())

    assert tuple(values) == tuple(name for name in SCALAR_CONTEXT_FIELDS if name in complete_context())
    assert values["MachineId"] == "machine-1"
    assert values["ActualSpindleRpm"] == 8500


@pytest.mark.parametrize("field", FEATURE_NAMES)
def test_missing_feature_is_rejected(field):
    context = complete_context()
    context.pop(field)

    if field in FEATURE_NAMES:
        with pytest.raises(ValueError, match="Missing required CNC features"):
            build_feature_vector(context)
    else:
        validate_context(context)


def test_none_is_unknown_and_is_rejected():
    context = complete_context()
    context["ActualFeedMmPerMin"] = None

    with pytest.raises(ValueError, match="Missing required CNC features"):
        build_feature_vector(context)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ActualSpindleRpm", -1),
        ("ActualFeedMmPerMin", float("nan")),
        ("SPowerPercent", float("inf")),
        ("ToolDiameterMm", -0.1),
        ("ToolTeethCount", True),
        ("ToolRemainingUsePercent", 101),
    ],
)
def test_invalid_feature_values_are_rejected(field, value):
    context = complete_context()
    context[field] = value

    with pytest.raises(ValueError):
        build_feature_vector(context)


def test_nested_objects_are_rejected_at_service_boundary():
    context = complete_context()
    context["Tool"] = {"name": "T01"}

    with pytest.raises(ValueError, match="scalar value"):
        validate_context(context)


def test_unknown_fields_are_rejected():
    context = complete_context()
    context["Unexpected"] = 1

    with pytest.raises(ValueError, match="Unknown MachiningContext fields"):
        validate_context(context)


def test_schema_is_json_safe_and_declares_missing_policy():
    schema = feature_schema()

    assert schema["version"] == 2
    assert schema["feature_names"] == list(FEATURE_NAMES)
    assert schema["scalar_context_fields"] == list(SCALAR_CONTEXT_FIELDS)
    assert schema["missing_value_policy"] == "nullable_context; reject_missing_model_features"
    assert schema["temporal_policy"] == "one_snapshot_aligned_to_one_audio_window"
