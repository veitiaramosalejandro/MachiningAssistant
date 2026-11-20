"""Typed representation of the payload received from the RTSP metadata track.

This module mirrors the observed ``x-machining-metadata`` payload. It is a
stream transport contract, not ``MachiningContext``. No field is renamed,
inferred, validated as a model feature, or filled with a default value here.

The ``quality`` fields are represented because they are present in the raw
payload, but they are annotations/output data and must not be used as model
inputs or training targets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


STREAM_FEATURE_SCHEMA_VERSION = 1
STREAM_FEATURE_NAMES = (
    "ProgrammedRpm",
    "SpindleAdjustmentPercent",
    "AdjustedRpm",
    "ProgrammedMmPerMin",
    "FeedAdjustmentPercent",
    "AdjustedMmPerMin",
    "ToolDiameterMm",
)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class StreamSpindleSpeed:
    ProgrammedRpm: Any = None
    AdjustmentPercent: Any = None
    AdjustedRpm: Any = None

    @classmethod
    def from_mapping(cls, value: object) -> "StreamSpindleSpeed":
        data = _mapping(value)
        return cls(data.get("ProgrammedRpm"), data.get("AdjustmentPercent"), data.get("AdjustedRpm"))

    def to_mapping(self) -> dict[str, Any]:
        return {"ProgrammedRpm": self.ProgrammedRpm, "AdjustmentPercent": self.AdjustmentPercent, "AdjustedRpm": self.AdjustedRpm}


@dataclass(frozen=True)
class StreamFeedRate:
    ProgrammedMmPerMin: Any = None
    AdjustmentPercent: Any = None
    AdjustedMmPerMin: Any = None

    @classmethod
    def from_mapping(cls, value: object) -> "StreamFeedRate":
        data = _mapping(value)
        return cls(data.get("ProgrammedMmPerMin"), data.get("AdjustmentPercent"), data.get("AdjustedMmPerMin"))

    def to_mapping(self) -> dict[str, Any]:
        return {"ProgrammedMmPerMin": self.ProgrammedMmPerMin, "AdjustmentPercent": self.AdjustmentPercent, "AdjustedMmPerMin": self.AdjustedMmPerMin}


@dataclass(frozen=True)
class StreamTool:
    Tool: Any = None
    Type: Any = None
    Material: Any = None
    HeightMm: Any = None
    DiameterMm: Any = None
    CornerRadiusMm: Any = None
    Flutes: Any = None
    RemainingUsePercent: Any = None
    UsageTimeAtProgramStartMinutes: Any = None

    @classmethod
    def from_mapping(cls, value: object) -> "StreamTool":
        data = _mapping(value)
        return cls(*(data.get(name) for name in (
            "Tool", "Type", "Material", "HeightMm", "DiameterMm",
            "CornerRadiusMm", "Flutes", "RemainingUsePercent",
            "UsageTimeAtProgramStartMinutes",
        )))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "Tool": self.Tool, "Type": self.Type, "Material": self.Material,
            "HeightMm": self.HeightMm, "DiameterMm": self.DiameterMm,
            "CornerRadiusMm": self.CornerRadiusMm, "Flutes": self.Flutes,
            "RemainingUsePercent": self.RemainingUsePercent,
            "UsageTimeAtProgramStartMinutes": self.UsageTimeAtProgramStartMinutes,
        }


@dataclass(frozen=True)
class StreamCncParameters:
    SpindleSpeed: StreamSpindleSpeed = StreamSpindleSpeed()
    FeedRate: StreamFeedRate = StreamFeedRate()
    StkXY: Any = None
    StkZ: Any = None
    IncXY: Any = None
    IncZ: Any = None

    @classmethod
    def from_mapping(cls, value: object) -> "StreamCncParameters":
        data = _mapping(value)
        return cls(
            SpindleSpeed=StreamSpindleSpeed.from_mapping(data.get("SpindleSpeed")),
            FeedRate=StreamFeedRate.from_mapping(data.get("FeedRate")),
            StkXY=data.get("StkXY"), StkZ=data.get("StkZ"),
            IncXY=data.get("IncXY"), IncZ=data.get("IncZ"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "SpindleSpeed": self.SpindleSpeed.to_mapping(), "FeedRate": self.FeedRate.to_mapping(),
            "StkXY": self.StkXY, "StkZ": self.StkZ, "IncXY": self.IncXY, "IncZ": self.IncZ,
        }


@dataclass(frozen=True)
class StreamCncContext:
    Machine: Any = None
    Operator: Any = None
    Work: Any = None
    Part: Any = None
    PartMaterial: Any = None
    Operation: Any = None
    Tool: StreamTool = StreamTool()
    Parameters: StreamCncParameters = StreamCncParameters()
    Notes: Any = None
    Strategy: Any = None

    @classmethod
    def from_mapping(cls, value: object) -> "StreamCncContext":
        data = _mapping(value)
        return cls(
            Machine=data.get("Machine"), Operator=data.get("Operator"),
            Work=data.get("Work"), Part=data.get("Part"),
            PartMaterial=data.get("PartMaterial"), Operation=data.get("Operation"),
            Tool=StreamTool.from_mapping(data.get("Tool")),
            Parameters=StreamCncParameters.from_mapping(data.get("Parameters")),
            Notes=data.get("Notes"), Strategy=data.get("Strategy"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "Machine": self.Machine, "Operator": self.Operator, "Work": self.Work,
            "Part": self.Part, "PartMaterial": self.PartMaterial, "Operation": self.Operation,
            "Tool": self.Tool.to_mapping(), "Parameters": self.Parameters.to_mapping(),
            "Notes": self.Notes, "Strategy": self.Strategy,
        }


@dataclass(frozen=True)
class StreamContextEnvelope:
    id: Any = None
    timestamp_seconds: Any = None
    context: StreamCncContext = StreamCncContext()

    @classmethod
    def from_mapping(cls, value: object) -> "StreamContextEnvelope":
        data = _mapping(value)
        return cls(data.get("id"), data.get("timestamp_seconds"), StreamCncContext.from_mapping(data.get("context")))

    def to_mapping(self) -> dict[str, Any]:
        return {"id": self.id, "timestamp_seconds": self.timestamp_seconds, "context": self.context.to_mapping()}


@dataclass(frozen=True)
class StreamSessionMetadata:
    StartTime: Any = None
    EndTime: Any = None
    DurationSeconds: Any = None
    Duration: Any = None
    Mode: Any = None
    VideoFile: Any = None
    AudioFile: Any = None
    SegmentMinutes: Any = None
    AudioVolumePercent: Any = None
    AudioBufferMilliseconds: Any = None
    VideoFiles: list[Any] | None = None
    AudioFiles: list[Any] | None = None
    Context: StreamCncContext = StreamCncContext()

    @classmethod
    def from_mapping(cls, value: object) -> "StreamSessionMetadata":
        data = _mapping(value)
        return cls(
            StartTime=data.get("StartTime"), EndTime=data.get("EndTime"),
            DurationSeconds=data.get("DurationSeconds"), Duration=data.get("Duration"),
            Mode=data.get("Mode"), VideoFile=data.get("VideoFile"), AudioFile=data.get("AudioFile"),
            SegmentMinutes=data.get("SegmentMinutes"), AudioVolumePercent=data.get("AudioVolumePercent"),
            AudioBufferMilliseconds=data.get("AudioBufferMilliseconds"),
            VideoFiles=data.get("VideoFiles"), AudioFiles=data.get("AudioFiles"),
            Context=StreamCncContext.from_mapping(data.get("Context")),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "StartTime": self.StartTime, "EndTime": self.EndTime,
            "DurationSeconds": self.DurationSeconds, "Duration": self.Duration,
            "Mode": self.Mode, "VideoFile": self.VideoFile, "AudioFile": self.AudioFile,
            "SegmentMinutes": self.SegmentMinutes, "AudioVolumePercent": self.AudioVolumePercent,
            "AudioBufferMilliseconds": self.AudioBufferMilliseconds,
            "VideoFiles": self.VideoFiles, "AudioFiles": self.AudioFiles,
            "Context": self.Context.to_mapping(),
        }


@dataclass(frozen=True)
class StreamCncEvent:
    id: Any = None
    timestamp_seconds: Any = None
    quality: Any = None  # Raw annotation; never a model input or training target.
    observations: Any = None
    observations_format: Any = None
    created_at_utc: Any = None
    updated_at_utc: Any = None
    parameters: StreamCncParameters = StreamCncParameters()
    machine_activity: Any = None
    machine_activity_label: Any = None
    occurrence: Any = None
    value: Any = None
    parameter_changes: dict[str, Any] | None = None
    event_changes: dict[str, Any] | None = None
    machine_state: Any = None
    machine_state_label: Any = None

    @classmethod
    def from_mapping(cls, value: object) -> "StreamCncEvent":
        data = _mapping(value)
        return cls(
            id=data.get("id"), timestamp_seconds=data.get("timestamp_seconds"),
            quality=data.get("quality"), observations=data.get("observations"),
            observations_format=data.get("observations_format"), created_at_utc=data.get("created_at_utc"),
            updated_at_utc=data.get("updated_at_utc"), parameters=StreamCncParameters.from_mapping(data.get("parameters")),
            machine_activity=data.get("machine_activity"), machine_activity_label=data.get("machine_activity_label"),
            occurrence=data.get("occurrence"), value=data.get("value"),
            parameter_changes=data.get("parameter_changes"), event_changes=data.get("event_changes"),
            machine_state=data.get("machine_state"), machine_state_label=data.get("machine_state_label"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "id": self.id, "timestamp_seconds": self.timestamp_seconds, "quality": self.quality,
            "observations": self.observations, "observations_format": self.observations_format,
            "created_at_utc": self.created_at_utc, "updated_at_utc": self.updated_at_utc,
            "parameters": self.parameters.to_mapping(), "machine_activity": self.machine_activity,
            "machine_activity_label": self.machine_activity_label, "occurrence": self.occurrence,
            "value": self.value, "parameter_changes": self.parameter_changes,
            "event_changes": self.event_changes, "machine_state": self.machine_state,
            "machine_state_label": self.machine_state_label,
        }


@dataclass(frozen=True)
class StreamCncMetadata:
    """Complete observed RTSP metadata message, without MachiningContext mapping."""

    type: Any = None
    recordingId: Any = None
    sequence: Any = None
    timestampSeconds: Any = None
    reason: Any = None
    timeReference: Any = None
    durationSeconds: Any = None
    metadata: StreamSessionMetadata | None = None
    context: StreamContextEnvelope | None = None
    parameters: StreamCncParameters | None = None
    quality: Any = None  # Raw payload annotation; excluded from model features.
    event: StreamCncEvent | None = None

    @classmethod
    def from_message(cls, message: Mapping[str, Any]) -> "StreamCncMetadata":
        data = _mapping(message)
        return cls(
            type=data.get("type"), recordingId=data.get("recordingId"), sequence=data.get("sequence"),
            timestampSeconds=data.get("timestampSeconds"), reason=data.get("reason"),
            timeReference=data.get("timeReference"), durationSeconds=data.get("durationSeconds"),
            metadata=StreamSessionMetadata.from_mapping(data["metadata"]) if isinstance(data.get("metadata"), Mapping) else None,
            context=StreamContextEnvelope.from_mapping(data["context"]) if isinstance(data.get("context"), Mapping) else None,
            parameters=StreamCncParameters.from_mapping(data["parameters"]) if isinstance(data.get("parameters"), Mapping) else None,
            quality=data.get("quality"),
            event=StreamCncEvent.from_mapping(data["event"]) if isinstance(data.get("event"), Mapping) else None,
        )

    def to_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name in ("type", "recordingId", "sequence", "timestampSeconds", "reason", "timeReference", "durationSeconds", "quality"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        if self.metadata is not None:
            result["metadata"] = self.metadata.to_mapping()
        if self.context is not None:
            result["context"] = self.context.to_mapping()
        if self.parameters is not None:
            result["parameters"] = self.parameters.to_mapping()
        if self.event is not None:
            result["event"] = self.event.to_mapping()
        return result


def _as_metadata(value: StreamCncMetadata | Mapping[str, Any]) -> StreamCncMetadata:
    return value if isinstance(value, StreamCncMetadata) else StreamCncMetadata.from_message(value)


def build_stream_feature_vector(value: StreamCncMetadata | Mapping[str, Any]) -> np.ndarray:
    """Build the raw CNC vector from fields actually sent by the RTSP stream.

    This is intentionally independent from ``MachiningContext``. The vector
    contains programmed/adjusted spindle and feed values plus tool diameter;
    it does not use ``quality`` or any predicted class.
    """
    metadata = _as_metadata(value)
    context = metadata.context.context if metadata.context is not None else StreamCncContext()
    parameters = metadata.parameters or context.Parameters
    tool = context.Tool
    values = [
        parameters.SpindleSpeed.ProgrammedRpm,
        parameters.SpindleSpeed.AdjustmentPercent,
        parameters.SpindleSpeed.AdjustedRpm,
        parameters.FeedRate.ProgrammedMmPerMin,
        parameters.FeedRate.AdjustmentPercent,
        parameters.FeedRate.AdjustedMmPerMin,
        tool.DiameterMm,
    ]
    missing = [name for name, item in zip(STREAM_FEATURE_NAMES, values) if item is None]
    if missing:
        raise ValueError(f"Missing RTSP CNC features: {', '.join(missing)}")
    try:
        vector = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("RTSP CNC features must be numeric") from exc
    if vector.shape != (len(STREAM_FEATURE_NAMES),) or not np.isfinite(vector).all():
        raise ValueError("RTSP CNC features must be finite numeric values")
    if np.any(vector < 0):
        raise ValueError("RTSP CNC features must be non-negative")
    return np.ascontiguousarray(vector)


class StreamFeatureNormalizer:
    """Training-only normalizer for the independent RTSP CNC feature schema."""

    def __init__(self, mean: np.ndarray | list[float], scale: np.ndarray | list[float]) -> None:
        self.feature_names = STREAM_FEATURE_NAMES
        self.mean = self._stats(mean, "mean")
        self.scale = self._stats(scale, "scale")
        if np.any(self.scale <= 0):
            raise ValueError("RTSP CNC normalizer scale values must be positive")

    @staticmethod
    def _stats(values: np.ndarray | list[float], name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape != (len(STREAM_FEATURE_NAMES),) or not np.isfinite(array).all():
            raise ValueError(f"RTSP CNC normalizer {name} has an invalid shape or value")
        return np.ascontiguousarray(array)

    @classmethod
    def fit(cls, vectors: np.ndarray | list[list[float]]) -> "StreamFeatureNormalizer":
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.ndim != 2 or matrix.shape[1] != len(STREAM_FEATURE_NAMES) or matrix.shape[0] == 0:
            raise ValueError(f"RTSP CNC training vectors must have shape (rows, {len(STREAM_FEATURE_NAMES)})")
        if not np.isfinite(matrix).all():
            raise ValueError("RTSP CNC training vectors must be finite")
        mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale == 0] = 1.0
        return cls(mean, scale)

    @classmethod
    def fit_metadata(cls, values: list[StreamCncMetadata | Mapping[str, Any]]) -> "StreamFeatureNormalizer":
        if not values:
            raise ValueError("At least one RTSP CNC training metadata record is required")
        return cls.fit(np.stack([build_stream_feature_vector(value) for value in values]))

    def transform(self, vectors: np.ndarray | list[float] | list[list[float]]) -> np.ndarray:
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim not in (1, 2) or array.shape[-1] != len(STREAM_FEATURE_NAMES):
            raise ValueError("RTSP CNC vector has an invalid shape")
        if not np.isfinite(array).all():
            raise ValueError("RTSP CNC vector must be finite")
        return np.ascontiguousarray((array - self.mean) / self.scale, dtype=np.float32)

    def transform_metadata(self, value: StreamCncMetadata | Mapping[str, Any]) -> np.ndarray:
        return self.transform(build_stream_feature_vector(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": "rtsp_stream",
            "feature_schema_version": STREAM_FEATURE_SCHEMA_VERSION,
            "feature_names": list(STREAM_FEATURE_NAMES),
            "feature_mean": self.mean.tolist(),
            "feature_scale": self.scale.tolist(),
            "missing_value_policy": "reject_missing_model_features",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StreamFeatureNormalizer":
        if payload.get("contract") != "rtsp_stream" or payload.get("feature_schema_version") != STREAM_FEATURE_SCHEMA_VERSION:
            raise ValueError("Serialized RTSP CNC normalizer contract is not supported")
        if list(payload.get("feature_names", ())) != list(STREAM_FEATURE_NAMES):
            raise ValueError("Serialized RTSP CNC feature order does not match the stream schema")
        return cls(payload["feature_mean"], payload["feature_scale"])


__all__ = [
    "StreamCncContext", "StreamCncEvent", "StreamCncMetadata", "StreamCncParameters",
    "StreamContextEnvelope", "StreamFeatureNormalizer", "StreamFeedRate", "StreamSessionMetadata",
    "StreamSpindleSpeed", "StreamTool", "STREAM_FEATURE_NAMES", "build_stream_feature_vector",
]
