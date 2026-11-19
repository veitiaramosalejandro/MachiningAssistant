import pytest

from Dataset.Preprocessing.cnc_temporal_alignment import align_context_window, window_timing


def base_context():
    return {"ContextId": "ctx-start", "ActualSpindleRpm": 8500}


def test_window_timing_uses_half_open_frame_range():
    timing = window_timing(44100, 132300, 44100)

    assert timing == {
        "start_frame": 44100,
        "end_frame_exclusive": 132300,
        "start_time": 1.0,
        "end_time": 3.0,
        "duration_seconds": 2.0,
    }


def test_constant_context_is_attached_to_complete_window():
    result = align_context_window(0, 220500, base_context(), sample_rate=44100)

    assert result["start_time"] == 0.0
    assert result["end_time"] == 5.0
    assert result["alignment_policy"] == "constant_per_window"
    assert result["context_change_count"] == 0
    assert result["context_spans"][0]["first_source_frame"] == 0


def test_event_at_window_start_replaces_initial_snapshot():
    result = align_context_window(
        220500,
        441000,
        base_context(),
        [{"effective_source_frame": 220500, "context": {"ContextId": "ctx-2"}}],
        sample_rate=44100,
    )

    assert result["context"]["ContextId"] == "ctx-2"
    assert result["context_start_frame"] == 220500


def test_event_at_exclusive_end_belongs_to_next_window():
    result = align_context_window(
        0,
        220500,
        base_context(),
        [{"effective_source_frame": 220500, "context": {"ContextId": "ctx-2"}}],
    )

    assert result["context"]["ContextId"] == "ctx-start"
    assert result["context_change_count"] == 0


def test_internal_change_is_rejected_for_initial_constant_policy():
    with pytest.raises(ValueError, match="inside the window"):
        align_context_window(
            0,
            220500,
            base_context(),
            [{"effective_source_frame": 100000, "context": {"ContextId": "ctx-2"}}],
        )


def test_internal_changes_can_be_returned_as_spans_explicitly():
    result = align_context_window(
        0,
        220500,
        base_context(),
        [{"effective_source_frame": 100000, "context": {"ContextId": "ctx-2"}}],
        allow_variable=True,
    )

    assert result["alignment_policy"] == "context_spans"
    assert [(item["first_source_frame"], item["end_source_frame_exclusive"])
            for item in result["context_spans"]] == [(0, 100000), (100000, 220500)]


def test_context_event_without_source_frame_is_rejected():
    with pytest.raises(ValueError, match="effective_source_frame"):
        align_context_window(
            0,
            220500,
            base_context(),
            [{"timestamp_seconds": 1, "context": {"ContextId": "ctx-2"}}],
        )


def test_watermark_must_cover_window():
    with pytest.raises(ValueError, match="watermark"):
        align_context_window(
            0,
            220500,
            base_context(),
            sample_rate=44100,
            context_complete_through_source_frame=220499,
        )


def test_duplicate_event_frames_are_rejected():
    with pytest.raises(ValueError, match="multiple context events"):
        align_context_window(
            0,
            220500,
            base_context(),
            [
                {"effective_source_frame": 100, "context": {"ContextId": "a"}},
                {"effective_source_frame": 100, "context": {"ContextId": "b"}},
            ],
        )
