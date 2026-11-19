"""Phase 3 and 4 tests for the CNC multimodal input contract."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from Dataset.Preprocessing.cnc_features import CncFeatureNormalizer, FEATURE_NAMES

CNN_DIR = Path(__file__).resolve().parents[1] / "CNN1D-VAE"
sys.path.insert(0, str(CNN_DIR))
from classifier import StateClassifier  # noqa: E402
from model import ModelConfig  # noqa: E402


def _vectors(rows=4):
    return np.arange(rows * len(FEATURE_NAMES), dtype=np.float32).reshape(rows, -1) + 1


def test_normalizer_fits_training_statistics_and_roundtrips_metadata():
    normalizer = CncFeatureNormalizer.fit(_vectors())
    transformed = normalizer.transform(_vectors())

    np.testing.assert_allclose(transformed.mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(transformed.std(axis=0), 1, atol=1e-6)
    restored = CncFeatureNormalizer.from_dict(normalizer.to_dict())
    np.testing.assert_array_equal(restored.transform(_vectors()), transformed)


def test_normalizer_rejects_wrong_shape_and_nonfinite_values():
    normalizer = CncFeatureNormalizer.fit(_vectors())
    with pytest.raises(ValueError, match="shape"):
        normalizer.transform(np.zeros(len(FEATURE_NAMES) - 1))
    with pytest.raises(ValueError, match="finite"):
        normalizer.transform(np.full(len(FEATURE_NAMES), np.nan))


def test_multimodal_classifier_fuses_audio_mean_with_context_embedding():
    model = StateClassifier(
        ModelConfig(sequence_length=64, latent_dim=4),
        num_classes=4,
        context_dim=len(FEATURE_NAMES),
        context_embedding_dim=6,
    ).eval()
    audio = torch.randn(2, 64, 2, dtype=torch.float32)
    context = torch.zeros(2, len(FEATURE_NAMES), dtype=torch.float32)
    with torch.no_grad():
        output = model(audio, reconstruct=False, context=context)

    assert output["mu"].shape == (2, 4)
    assert output["context_embedding"].shape == (2, 6)
    assert output["logits"].shape == (2, 4)
    assert all(torch.isfinite(value).all() for value in output.values())


def test_multimodal_classifier_requires_valid_context():
    model = StateClassifier(ModelConfig(sequence_length=64), context_dim=7).eval()
    audio = torch.randn(2, 64, 2, dtype=torch.float32)
    with pytest.raises(ValueError, match="context is required"):
        model(audio, reconstruct=False)
    with pytest.raises(ValueError, match="Expected context shape"):
        model(audio, context=torch.zeros(2, 6), reconstruct=False)
    with pytest.raises(TypeError, match="float32"):
        model(audio, context=torch.zeros(2, 7, dtype=torch.float64), reconstruct=False)
    with pytest.raises(ValueError, match="finite"):
        context = torch.zeros(2, 7)
        context[0, 0] = float("nan")
        model(audio, context=context, reconstruct=False)


