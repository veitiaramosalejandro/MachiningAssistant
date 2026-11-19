"""Deterministic quality labels for the RTSP training stream."""
from __future__ import annotations


QUALITY_LABELS = (
    "broken",
    "very_bad",
    "bad",
    "sufficient",
    "good",
    "excellent",
)

QUALITY_SCORES = {
    "broken": 0.0,
    "very_bad": 0.2,
    "bad": 0.4,
    "sufficient": 0.6,
    "good": 0.8,
    "excellent": 1.0,
}


def quality_to_label(value: object) -> tuple[str, float]:
    """Convert metadata quality 0..5 into a stable class and score."""
    if isinstance(value, bool):
        raise ValueError("quality debe ser un entero entre 0 y 5")
    try:
        quality = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("quality debe ser un entero entre 0 y 5") from exc
    if quality < 0 or quality > 5 or quality != float(value):
        raise ValueError("quality debe ser un entero entre 0 y 5")
    label = QUALITY_LABELS[quality]
    return label, QUALITY_SCORES[label]


__all__ = ["QUALITY_LABELS", "QUALITY_SCORES", "quality_to_label"]
