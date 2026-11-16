"""Checkpoint preprocessing and deterministic supervised inference checks."""
from dataclasses import asdict
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classifier import StateClassifier
from model import ModelConfig
from state_runtime import MODEL_FAMILY, StateEngine


CLASS_NAMES = ["normal_cutting", "abnormal_cutting", "defective_tool_normal_cutting"]


@pytest.fixture
def saved_checkpoint(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(23)
    config = ModelConfig(sequence_length=64)
    model = StateClassifier(config).eval()
    state = {
        "model_family": MODEL_FAMILY,
        "model_config": asdict(config),
        "state_dict": model.state_dict(),
        "normalizer": {"mean": [17., -32.], "scale": [120., 1500.]},
        "preprocessing": {"target_sample_rate_hz": 48000, "sequence_length": 64},
        "class_names": CLASS_NAMES,
        "status": "msm_supervised_experimental",
    }
    path = tmp_path / "best_classifier.pt"
    torch.save(state, path)
    return path, model, state


def physical_windows():
    values = np.random.default_rng(5).normal(size=(2, 64, 2)).astype(np.float32)
    return values * np.array([200., 3000.], np.float32) + np.array([20., -50.], np.float32)


def test_checkpoint_reuses_normalizer_and_restores_logits(saved_checkpoint):
    path, model, state = saved_checkpoint
    signal = physical_windows()
    mean = np.asarray(state["normalizer"]["mean"], np.float32)
    scale = np.asarray(state["normalizer"]["scale"], np.float32)
    with torch.inference_mode():
        logits = model(torch.from_numpy((signal-mean)/scale), reconstruct=False)["logits"]
        expected_scores = torch.softmax(logits, dim=-1).numpy()
    engine = StateEngine.from_checkpoint(path)
    result = engine.predict_windows(signal)
    np.testing.assert_allclose(result["logits"], logits.numpy(), rtol=0, atol=0)
    np.testing.assert_allclose(result["class_scores"], expected_scores, rtol=0, atol=0)
    assert result["predicted_label"] == expected_scores.argmax(axis=1).tolist()
    assert result["predicted_state"] == [CLASS_NAMES[i] for i in result["predicted_label"]]
    assert result["domain"] == "MSM imi_vm20i"
    assert result["scores_calibrated"] is False
    assert result["validated_on_company_data"] is False


def test_single_window_is_deterministic_and_decoder_is_not_called(saved_checkpoint):
    engine = StateEngine.from_checkpoint(saved_checkpoint[0])
    signal = physical_windows()[0]
    with patch.object(engine.model.vae.decoder, "forward", side_effect=AssertionError("Decoder called")):
        first = engine.predict_windows(signal)
        second = engine.predict_windows(signal[None])
    assert len(first["predicted_label"]) == 1
    assert first["class_scores"] == second["class_scores"]
    np.testing.assert_allclose(np.sum(first["class_scores"], axis=1), [1.], atol=1e-6)


def test_rate_shape_nonfinite_and_raw_integer_audio_are_rejected(saved_checkpoint):
    engine = StateEngine.from_checkpoint(saved_checkpoint[0])
    signal = physical_windows()
    with pytest.raises(ValueError, match="48000"):
        engine.predict_windows(signal, sample_rate=44100)
    with pytest.raises(ValueError, match="Expected nonempty"):
        engine.predict_windows(signal[:, :32])
    with pytest.raises(ValueError, match="Expected nonempty"):
        engine.predict_windows(signal.transpose(0, 2, 1))
    with pytest.raises(TypeError, match="Acc0"):
        engine.predict_windows(signal.astype(np.int16))
    signal[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        engine.predict_windows(signal)


def test_checkpoint_family_and_preprocessing_contract_are_checked(saved_checkpoint):
    path, _, state = saved_checkpoint
    state["model_family"] = "another_architecture"
    torch.save(state, path)
    with pytest.raises(ValueError, match="classification checkpoint"):
        StateEngine.from_checkpoint(path)
    state["model_family"] = MODEL_FAMILY
    state["preprocessing"]["target_sample_rate_hz"] = 16000
    torch.save(state, path)
    with pytest.raises(ValueError, match="48000"):
        StateEngine.from_checkpoint(path)
