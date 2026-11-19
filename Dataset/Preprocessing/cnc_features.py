"""Validated CNC context and numeric features for the CNN1D-VAE branch.

This module accepts every scalar JSON property declared by
MachiningMonitor.Contracts.MachiningContext. Nested objects and arrays are
rejected. The numeric model vector remains a separate, deterministic subset:
identifiers, free text and timestamps are retained in the validated transport
context but are not silently converted to arbitrary numbers for the CNN.

The module does not read files, infer labels, or modify the audio tensor. A
caller must first align one context snapshot with one audio window and may
only pass the resulting context to ``build_feature_vector``.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np


FEATURE_SCHEMA_VERSION = 2

# Scalar properties from MachiningContext.cs. Nullable properties may be None;
# required transport identity fields are checked separately below. This list
# is the service boundary, not a claim that every field is a safe ML feature.
SCALAR_CONTEXT_FIELDS = (
    "SchemaVersion",
    "Simulation",
    "ContextId",
    "MachineId",
    "DriverSessionId",
    "Sequence",
    "OccurredAtUtc",
    "ExecutionId",
    "ProgramName",
    "ProgramPath",
    "MachineName",
    "Operator",
    "Work",
    "Part",
    "PartMaterial",
    "PartMaterialDesignation",
    "Operation",
    "Strategy",
    "Tool",
    "ToolType",
    "ToolTypeDesignation",
    "ToolMaterial",
    "ToolMaterialDesignation",
    "ToolHeightMm",
    "ToolDiameterMm",
    "ToolCornerRadiusMm",
    "ToolTeethCount",
    "ToolRemainingUsePercent",
    "ToolUsageTimeAtProgramStartMinutes",
    "StockXYMm",
    "StockZMm",
    "IncrementXYMm",
    "IncrementZMm",
    "ProgrammedSpindleRpm",
    "SpindleOverridePercent",
    "AdjustedSpindleRpm",
    "ProgrammedFeedMmPerMin",
    "FeedOverridePercent",
    "AdjustedFeedMmPerMin",
    "ActualSpindleRpm",
    "ActualFeedMmPerMin",
    "SPowerPercent",
    "WPowerPercent",
    "Notes",
)

_REQUIRED_CONTEXT_FIELDS = frozenset({
    "SchemaVersion",
    "ContextId",
    "MachineId",
    "DriverSessionId",
    "OccurredAtUtc",
})

# Keep this order stable. It is part of the model/checkpoint contract.
FEATURE_NAMES = (
    "ActualSpindleRpm",
    "ActualFeedMmPerMin",
    "SPowerPercent",
    "WPowerPercent",
    "ToolDiameterMm",
    "ToolTeethCount",
    "ToolRemainingUsePercent",
)

FEATURE_UNITS = {
    "ActualSpindleRpm": "rpm",
    "ActualFeedMmPerMin": "mm/min",
    "SPowerPercent": "%",
    "WPowerPercent": "%",
    "ToolDiameterMm": "mm",
    "ToolTeethCount": "count",
    "ToolRemainingUsePercent": "%",
}

# Hard limits only where the contract explicitly defines a physical range.
# RPM, feed, power and geometry are checked as finite non-negative values.
_NON_NEGATIVE_FEATURES = frozenset({
    "ActualSpindleRpm",
    "ActualFeedMmPerMin",
    "SPowerPercent",
    "WPowerPercent",
    "ToolDiameterMm",
    "ToolTeethCount",
})


def feature_schema() -> dict:
    """Return the transport and model feature schema for metadata."""
    return {
        "version": FEATURE_SCHEMA_VERSION,
        "scalar_context_fields": list(SCALAR_CONTEXT_FIELDS),
        "feature_names": list(FEATURE_NAMES),
        "units": dict(FEATURE_UNITS),
        "missing_value_policy": "nullable_context; reject_missing_model_features",
        "source_contract": "MachiningMonitor.Contracts.MachiningContext",
        "temporal_policy": "one_snapshot_aligned_to_one_audio_window",
    }


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric value") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite numeric value")
    if name in _NON_NEGATIVE_FEATURES and number < 0:
        raise ValueError(f"{name} must be non-negative")
    if name == "ToolRemainingUsePercent" and not 0 <= number <= 100:
        raise ValueError("ToolRemainingUsePercent must be between 0 and 100")
    return number


def validate_context(context: Mapping[str, object]) -> dict[str, object]:
    """Validate and return all known scalar context fields.

    ``None`` is preserved for nullable MachiningContext properties. Required
    transport identity fields must be present and non-null. Nested objects and
    arrays are rejected rather than flattened implicitly.
    """
    if not isinstance(context, Mapping):
        raise TypeError("CNC context must be a mapping")

    unknown = sorted(set(context) - set(SCALAR_CONTEXT_FIELDS))
    if unknown:
        raise ValueError(f"Unknown MachiningContext fields: {', '.join(unknown)}")

    missing = [
        name for name in _REQUIRED_CONTEXT_FIELDS
        if name not in context or context[name] is None
    ]
    if missing:
        raise ValueError(f"Missing required MachiningContext fields: {', '.join(sorted(missing))}")

    for name, value in context.items():
        if isinstance(value, (Mapping, list, tuple, set)):
            raise ValueError(f"{name} must be a scalar value, not a nested object or array")

    return {name: context.get(name) for name in SCALAR_CONTEXT_FIELDS if name in context}


def build_feature_vector(context: Mapping[str, object]) -> np.ndarray:
    """Return the validated raw CNC vector as contiguous float32 values.

    The result is raw, unit-bearing data. Training code must fit and persist a
    separate normalizer using training rows only before passing it to a model.
    """
    values = validate_context(context)
    missing = [name for name in FEATURE_NAMES if values.get(name) is None]
    if missing:
        raise ValueError(f"Missing required CNC features: {', '.join(missing)}")
    return np.ascontiguousarray(
        np.asarray([_finite_number(name, values[name]) for name in FEATURE_NAMES], dtype=np.float32)
    )


__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "SCALAR_CONTEXT_FIELDS",
    "FEATURE_NAMES",
    "FEATURE_UNITS",
    "feature_schema",
    "validate_context",
    "build_feature_vector",
]
