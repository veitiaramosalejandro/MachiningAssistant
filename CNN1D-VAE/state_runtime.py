"""MSM state-classification inference with the checkpoint's preprocessing.

The input is an already time-aligned 48 kHz floating-point array. Its columns
must be Sound0 microphone PCM16 counts and Acc0 acceleration in mm/s^2.
This module does not read stereo WAVs, infer sensor units, or resample signals.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from device_support import resolve_device
from classifier import StateClassifier
from model import ModelConfig


MODEL_FAMILY = "CNN1D-VAE"
SAMPLE_RATE = 48000
DOMAIN = "MSM imi_vm20i"
CHANNEL_UNITS = ["Sound0 PCM16 counts", "Acc0 mm/s^2"]


class StateEngine:
    """Predict three supervised MSM state classes using posterior-mean logits.

    A single window retains a batch dimension of size one in returned lists.
    Softmax scores are not calibrated probabilities, and the model has not
    been validated on the company's recordings.
    """

    def __init__(self, model, normalizer, preprocessing, class_names, status, device='auto'):
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.config = self.model.config
        self.preprocessing = dict(preprocessing)
        self.class_names = list(class_names)
        self.status = status
        if self.config.input_channels != 2 or len(self.class_names) != 3:
            raise ValueError("MSM state inference requires two input channels and three classes")
        if self.preprocessing["target_sample_rate_hz"] != SAMPLE_RATE:
            raise ValueError(f"Checkpoint must use {SAMPLE_RATE} Hz preprocessing")
        if self.preprocessing["sequence_length"] != self.config.sequence_length:
            raise ValueError("Checkpoint model and preprocessing window lengths differ")
        self.mean = np.asarray(normalizer["mean"], dtype=np.float32)
        self.scale = np.asarray(normalizer["scale"], dtype=np.float32)
        if self.mean.shape != (2,) or self.scale.shape != (2,):
            raise ValueError("Checkpoint normalization must contain two channel means and scales")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError("Checkpoint normalization must be finite with positive scales")

    @classmethod
    def from_checkpoint(cls, path, device='auto'):
        """Load the best_classifier.pt format produced by experiment.py."""
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        if checkpoint["model_family"] != MODEL_FAMILY:
            raise ValueError(f"Expected a {MODEL_FAMILY} classification checkpoint")
        config = ModelConfig(**checkpoint["model_config"])
        model = StateClassifier(config)
        model.load_state_dict(checkpoint["state_dict"])
        return cls(model, checkpoint["normalizer"], checkpoint["preprocessing"],
                   checkpoint["class_names"], checkpoint.get("status", "unspecified"), device=device)

    def predict_windows(self, signal, sample_rate=SAMPLE_RATE):
        """Classify (B,T,2) or (T,2) aligned physical inputs without refitting.

        T must equal the saved model window length (2048 in the supplied
        trained models). Raw two-channel PCM audio is not a replacement for
        an independently measured and calibrated Acc0 acceleration channel.
        """
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz aligned input; no automatic resampling")
        values = np.asarray(signal)
        if not np.issubdtype(values.dtype, np.floating):
            raise TypeError("Supply floating-point Sound0 PCM counts and Acc0 mm/s^2, not a stereo PCM integer array")
        if values.ndim == 2:
            values = values[None]
        expected = (self.config.sequence_length, 2)
        if values.ndim != 3 or tuple(values.shape[1:]) != expected or values.shape[0] == 0:
            raise ValueError(f"Expected nonempty (batch, {expected[0]}, 2) or ({expected[0]}, 2) input")
        if not np.isfinite(values).all():
            raise ValueError("Input signal contains non-finite values")
        values = np.asarray(values, dtype=np.float32)
        normalized = (values - self.mean) / self.scale
        with torch.inference_mode():
            output = self.model(torch.from_numpy(np.ascontiguousarray(normalized)).to(self.device), reconstruct=False)
            logits = output["logits"]
            if not torch.isfinite(logits).all():
                raise RuntimeError("Classifier produced non-finite logits")
            scores = torch.softmax(logits, dim=-1).cpu().numpy()
        labels = scores.argmax(axis=-1).tolist()
        return {
            "model_family": MODEL_FAMILY, "device": str(self.device),
            "model_status": self.status,
            "predicted_label": labels,
            "predicted_state": [self.class_names[label] for label in labels],
            "class_names": self.class_names,
            "class_scores": scores.tolist(),
            "logits": logits.cpu().tolist(),
            "score_type": "softmax_class_scores",
            "scores_calibrated": False,
            "domain": DOMAIN,
            "validated_on_company_data": False,
            "sample_rate_hz": SAMPLE_RATE,
            "window_samples": self.config.sequence_length,
            "window_duration_s": self.config.sequence_length / SAMPLE_RATE,
            "input_channel_units": CHANNEL_UNITS,
        }
