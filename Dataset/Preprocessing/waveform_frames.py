"""Group consecutive waveform samples without resampling or feature reduction.

A frame vector keeps time-major/channel-interleaved sample order. A short final
frame is zero-padded internally; unframing removes only that final padding.
Both functions preserve the input tensor's dtype, device, and autograd graph.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def _positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def frame_waveform(x: torch.Tensor, frame_size: int) -> torch.Tensor:
    """Map [batch,time,channels] to [batch,ceil(time/frame_size),frame_size*channels]."""
    _positive_integer(frame_size, 'frame_size')
    if x.ndim != 3 or x.shape[1] < 1 or x.shape[2] < 1:
        raise ValueError('Expected waveform shape [batch, time, channels] with nonempty time and channels')
    batch, length, channels = x.shape
    steps = (length+frame_size-1)//frame_size
    padding = steps*frame_size-length
    if padding:
        x = F.pad(x, (0, 0, 0, padding))
    return x.reshape(batch, steps, frame_size*channels)


def unframe_waveform(
    frames: torch.Tensor, frame_size: int, original_length: int, channels: int = 2,
) -> torch.Tensor:
    """Restore [batch,original_length,channels], discarding only internal tail padding."""
    _positive_integer(frame_size, 'frame_size')
    _positive_integer(original_length, 'original_length')
    _positive_integer(channels, 'channels')
    expected_steps = (original_length+frame_size-1)//frame_size
    if frames.ndim != 3 or tuple(frames.shape[1:]) != (expected_steps, frame_size*channels):
        raise ValueError(
            f'Expected framed shape [batch, {expected_steps}, {frame_size*channels}] '
            f'for original_length={original_length}'
        )
    return frames.reshape(frames.shape[0], expected_steps*frame_size, channels)[:, :original_length].contiguous()
