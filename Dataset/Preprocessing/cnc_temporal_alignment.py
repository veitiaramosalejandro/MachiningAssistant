"""Frame-clock alignment between audio windows and service context snapshots.

The audio source frame is the authoritative timeline. Transport arrival time
and unrelated wall clocks are never used to decide which CNC context belongs
to a window. Context changes must carry ``effective_source_frame``; otherwise
the service cannot prove alignment.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import math


def _int_field(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if isinstance(value, str) and str(result) != value.strip():
        raise ValueError(f"{name} must be an integer")
    return result


def _finite_frame(name: str, value: object) -> int:
    frame = _int_field(name, value)
    if frame < 0:
        raise ValueError(f"{name} must be non-negative")
    return frame


def window_timing(
    start_frame: object,
    end_frame_exclusive: object,
    sample_rate: object,
) -> dict[str, float | int]:
    """Validate a half-open frame range and derive its seconds range."""
    start = _finite_frame("start_frame", start_frame)
    end = _finite_frame("end_frame_exclusive", end_frame_exclusive)
    try:
        rate = float(sample_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("sample_rate must be finite and positive") from exc
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("sample_rate must be finite and positive")
    if end <= start:
        raise ValueError("end_frame_exclusive must be greater than start_frame")
    return {
        "start_frame": start,
        "end_frame_exclusive": end,
        "start_time": start / rate,
        "end_time": end / rate,
        "duration_seconds": (end - start) / rate,
    }


def _event_frame(event: Mapping[str, object]) -> int:
    if "effective_source_frame" not in event or event["effective_source_frame"] is None:
        raise ValueError("context event lacks effective_source_frame")
    return _finite_frame("effective_source_frame", event["effective_source_frame"])


def _context_for_event(event: Mapping[str, object]) -> Mapping[str, object]:
    value = event.get("context")
    if not isinstance(value, Mapping):
        raise ValueError("context event must contain a context mapping")
    return value


def _sorted_events(context_events: Iterable[Mapping[str, object]]) -> list[tuple[int, Mapping[str, object]]]:
    indexed = []
    for index, event in enumerate(context_events):
        if not isinstance(event, Mapping):
            raise ValueError(f"context event {index} must be a mapping")
        indexed.append((_event_frame(event), event))
    indexed.sort(key=lambda item: item[0])
    for previous, current in zip(indexed, indexed[1:]):
        if previous[0] == current[0]:
            raise ValueError(f"multiple context events share source frame {current[0]}")
    return indexed


def align_context_window(
    start_frame: object,
    end_frame_exclusive: object,
    context_at_start: Mapping[str, object] | None,
    context_events: Iterable[Mapping[str, object]] = (),
    *,
    sample_rate: object = 44100,
    context_complete_through_source_frame: object | None = None,
    allow_variable: bool = False,
) -> dict:
    """Align a context snapshot/events to one half-open audio window.

    Events at ``start_frame`` apply to this window. Events at the exclusive end
    belong to the following window. With ``allow_variable=False`` (the initial
    model policy), an event strictly inside the window is rejected because one
    CNC vector cannot represent two contexts. With ``allow_variable=True``,
    context spans are returned for a later temporal-feature implementation.
    """
    timing = window_timing(start_frame, end_frame_exclusive, sample_rate)
    start = int(timing["start_frame"])
    end = int(timing["end_frame_exclusive"])
    events = _sorted_events(context_events)

    if context_at_start is not None and not isinstance(context_at_start, Mapping):
        raise ValueError("context_at_start must be a mapping or None")

    applicable = [(frame, event) for frame, event in events if frame <= start]
    inside = [(frame, event) for frame, event in events if start < frame < end]
    if applicable:
        selected_context = dict(_context_for_event(applicable[-1][1]))
        selected_frame = applicable[-1][0]
    elif context_at_start is not None:
        selected_context = dict(context_at_start)
        selected_frame = start
    else:
        raise ValueError("no context is available at the start of the audio window")

    if inside and not allow_variable:
        raise ValueError(
            "context changes inside the window; one constant CNC vector is not valid"
        )

    boundaries = [(start, selected_context)]
    for frame, event in inside:
        boundaries.append((frame, dict(_context_for_event(event))))

    spans = []
    for index, (span_start, context) in enumerate(boundaries):
        span_end = boundaries[index + 1][0] if index + 1 < len(boundaries) else end
        spans.append({
            "first_source_frame": span_start,
            "end_source_frame_exclusive": span_end,
            "start_time": span_start / float(sample_rate),
            "end_time": span_end / float(sample_rate),
            "context": context,
        })

    watermark = None
    if context_complete_through_source_frame is not None:
        watermark = _finite_frame(
            "context_complete_through_source_frame",
            context_complete_through_source_frame,
        )
        if watermark < end:
            raise ValueError(
                "context watermark does not cover the complete audio window"
            )

    result = {
        **timing,
        "alignment_policy": "constant_per_window" if not inside else "context_spans",
        "context_alignment_complete": watermark is not None or bool(context_at_start),
        "context_start_frame": selected_frame,
        "context": selected_context,
        "context_spans": spans,
        "context_change_count": len(inside),
    }
    return result


__all__ = ["window_timing", "align_context_window"]
