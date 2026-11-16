"""Lossless time/channel framing, tail padding, and gradient preservation."""
from pathlib import Path
import sys

import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from Dataset.Preprocessing.waveform_frames import frame_waveform, unframe_waveform


def test_frame_layout_keeps_time_major_channel_interleaving():
    x = torch.tensor([[[1., 10.], [2., 20.], [3., 30.], [4., 40.], [5., 50.]]])
    framed = frame_waveform(x, 2)
    expected = torch.tensor([[[1., 10., 2., 20.], [3., 30., 4., 40.], [5., 50., 0., 0.]]])
    torch.testing.assert_close(framed, expected, rtol=0, atol=0)
    restored = unframe_waveform(framed, 2, original_length=5, channels=2)
    torch.testing.assert_close(restored, x, rtol=0, atol=0)


@pytest.mark.parametrize('frame_size', [1, 2, 4, 8])
def test_noncontiguous_batched_input_round_trip_is_exact(frame_size):
    x = torch.arange(2*7*4, dtype=torch.float64).reshape(2, 7, 4)[:, :, ::2]
    assert not x.is_contiguous()
    framed = frame_waveform(x, frame_size)
    assert framed.shape == (2, (7+frame_size-1)//frame_size, frame_size*2)
    assert framed.dtype == x.dtype and framed.device == x.device
    actual = unframe_waveform(framed, frame_size, original_length=7, channels=2)
    torch.testing.assert_close(actual, x, rtol=0, atol=0)


def test_partial_last_frame_retains_valid_gradients_and_excludes_padding():
    x = torch.arange(10, dtype=torch.float64).reshape(1, 5, 2).requires_grad_()
    weights = torch.arange(1, 11, dtype=torch.float64).reshape_as(x)
    framed = frame_waveform(x, 3)
    framed.retain_grad()
    restored = unframe_waveform(framed, 3, original_length=5, channels=2)
    (restored*weights).sum().backward()
    torch.testing.assert_close(x.grad, weights, rtol=0, atol=0)
    assert framed.shape == (1, 2, 6)
    torch.testing.assert_close(framed[0, -1, -2:], torch.zeros(2, dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(framed.grad[0, -1, -2:], torch.zeros(2, dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(framed.grad[0, -1, :4], weights.reshape(-1)[6:], rtol=0, atol=0)


def test_unframe_rejects_a_shape_that_would_drop_or_invent_samples():
    with pytest.raises(ValueError, match='Expected framed shape'):
        unframe_waveform(torch.zeros(1, 1, 6), 3, original_length=5, channels=2)


@pytest.mark.parametrize('frame_size', [0, -1, True, 1.5])
def test_frame_size_must_be_a_positive_integer(frame_size):
    with pytest.raises(ValueError, match='positive integer'):
        frame_waveform(torch.zeros(1, 4, 2), frame_size)
