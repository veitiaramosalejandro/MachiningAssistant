"""Adapters for playback metadata to the validated CNN1D-VAE CNC contract."""
from __future__ import annotations

from datetime import datetime, timezone
import re

from Dataset.Preprocessing.cnc_features import validate_context


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _first(mapping, *names):
    for name in names:
        value = mapping.get(name)
        if value is not None:
            return value
    return None


def playback_message_to_context(message: dict, occurred_at_utc: str | None = None) -> dict:
    """Convert one playback snapshot/event into canonical scalar context.

    Missing model features are intentionally preserved as ``None`` so the
    feature validator can reject the sample instead of inventing zeros.
    """
    outer = _mapping(message)
    source = _mapping(outer.get("context"))
    parameters = _mapping(outer.get("parameters")) or _mapping(source.get("Parameters"))
    spindle = _mapping(parameters.get("SpindleSpeed"))
    feed = _mapping(parameters.get("FeedRate"))
    tool = _mapping(source.get("Tool"))
    event = _mapping(outer.get("event"))
    event_parameters = _mapping(event.get("parameters"))
    if event_parameters:
        parameters = event_parameters
        spindle = _mapping(parameters.get("SpindleSpeed"))
        feed = _mapping(parameters.get("FeedRate"))
    tool_name = _first(tool, "Tool", "Name")
    diameter = _first(tool, "DiameterMm", "ToolDiameterMm")
    if diameter is None and isinstance(tool_name, str):
        match = re.search(r"(?:^|\s)D(\d+(?:[.,]\d+)?)\b", tool_name, re.IGNORECASE)
        if match:
            diameter = float(match.group(1).replace(",", "."))
    timestamp = occurred_at_utc or datetime.now(timezone.utc).isoformat()
    recording_id = str(outer.get("recordingId") or "stream")
    sequence = outer.get("sequence", 0)
    context = {
        "SchemaVersion": 2,
        "ContextId": f"{recording_id}:{sequence}",
        "MachineId": str(_first(source, "MachineId", "Machine") or "unknown-machine"),
        "DriverSessionId": recording_id,
        "OccurredAtUtc": timestamp,
        "MachineName": _first(source, "MachineName", "Machine"),
        "Operator": source.get("Operator"), "Work": source.get("Work"),
        "Part": source.get("Part"), "PartMaterial": source.get("PartMaterial"),
        "Operation": source.get("Operation"), "Strategy": source.get("Strategy"),
        "Tool": tool_name, "ToolType": _first(tool, "Type", "ToolType"),
        "ToolMaterial": tool.get("Material"), "ToolDiameterMm": diameter,
        "ToolTeethCount": _first(tool, "Flutes", "ToolTeethCount"),
        "ToolRemainingUsePercent": _first(tool, "RemainingUsePercent", "ToolRemainingUsePercent"),
        "ToolHeightMm": _first(tool, "HeightMm", "ToolHeightMm"),
        "ToolCornerRadiusMm": _first(tool, "CornerRadiusMm", "ToolCornerRadiusMm"),
        "ActualSpindleRpm": _first(parameters, "ActualSpindleRpm") or _first(spindle, "ActualRpm"),
        "ActualFeedMmPerMin": _first(parameters, "ActualFeedMmPerMin") or _first(feed, "ActualMmPerMin"),
        "SPowerPercent": _first(parameters, "SPowerPercent", "SPower"),
        "WPowerPercent": _first(parameters, "WPowerPercent", "WPower"),
        "ProgrammedSpindleRpm": spindle.get("ProgrammedRpm"),
        "AdjustedSpindleRpm": spindle.get("AdjustedRpm"),
        "ProgrammedFeedMmPerMin": feed.get("ProgrammedMmPerMin"),
        "AdjustedFeedMmPerMin": feed.get("AdjustedMmPerMin"),
        "Notes": source.get("Notes"),
    }
    return validate_context(context)


__all__ = ["playback_message_to_context"]
