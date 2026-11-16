"""Supervised gradients, decoder bypass, and stable classifier inference."""
import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classifier import StateClassifier, classification_loss
from model import ModelConfig


torch.set_num_threads(1)


def signal(batch=3, length=64):
    t = torch.arange(length, dtype=torch.float32) / length
    frequencies = torch.arange(1, batch + 1, dtype=torch.float32)[:, None]
    return torch.stack((torch.sin(2 * torch.pi * frequencies * t),
                        0.4 * torch.cos(2 * torch.pi * (frequencies + 2) * t)), dim=-1)


def test_default_shape_has_three_class_logits_and_reconstruction():
    model = StateClassifier().eval()
    with torch.no_grad():
        output = model(signal(length=2048))
    assert output["logits"].shape == (3, 3)
    assert output["reconstruction"].shape == (3, 2048, 2)
    assert output["mu"].shape == (3, 16)
    assert all(torch.isfinite(value).all() for value in output.values())


def test_encoder_only_skips_decoder_and_encodes_once():
    model = StateClassifier(ModelConfig(sequence_length=64)).eval()
    x = signal()
    with torch.no_grad():
        reference = model(x)
    with patch.object(model.vae.encoder, "forward", wraps=model.vae.encoder.forward) as encode:
        with patch.object(model.vae.decoder, "forward", side_effect=AssertionError("Decoder called")):
            with patch.object(model.vae.decoder_input, "forward", side_effect=AssertionError("Decoder projection called")):
                output = model(x, reconstruct=False)
    assert encode.call_count == 1
    assert "reconstruction" not in output
    torch.testing.assert_close(output["logits"], reference["logits"], rtol=0, atol=0)
    losses = classification_loss(output, torch.tensor([0, 1, 2]))
    assert losses["reconstruction_loss"].item() == 0


def test_classification_gradients_reach_mean_encoder_and_head():
    torch.manual_seed(8)
    model = StateClassifier(ModelConfig(sequence_length=64))
    losses = classification_loss(model(signal(), reconstruct=False), torch.tensor([0, 1, 2]), beta=0)
    losses["ce_loss"].backward()
    for parameter in [model.vae.encoder[0].weight, model.vae.mu_head.weight,
                      model.classifier[0].weight, model.classifier[-1].weight]:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert model.vae.decoder[-1].weight.grad is None


def test_joint_reconstruction_reaches_variance_and_decoder():
    torch.manual_seed(9)
    model = StateClassifier(ModelConfig(sequence_length=64)).train()
    x = signal()
    with patch.object(model.vae.encoder, "forward", wraps=model.vae.encoder.forward) as encode:
        output = model(x, reconstruct=True)
    assert encode.call_count == 1
    losses = classification_loss(output, torch.tensor([0, 1, 2]), x)
    gradients = torch.autograd.grad(losses["reconstruction_loss"],
        [model.vae.logvar_head.weight, model.vae.decoder[-1].weight], retain_graph=True)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    losses["loss"].backward()
    assert model.classifier[-1].weight.grad.abs().sum() > 0


def test_loss_combines_ce_mse_and_independently_weighted_kl():
    logits = torch.tensor([[2., 0., -1.], [-1., 0., 2.]])
    target = torch.tensor([0, 2])
    x = torch.zeros(2, 64, 2)
    output = {"logits": logits, "reconstruction": torch.ones_like(x),
              "mu": torch.ones(2, 3), "logvar": torch.zeros(2, 3)}
    weights = torch.tensor([1., 2., 3.])
    losses = classification_loss(output, target, x, reconstruction_weight=0.2,
                                 beta=0.3, class_weights=weights)
    expected_ce = F.cross_entropy(logits, target, weight=weights)
    torch.testing.assert_close(losses["ce_loss"], expected_ce)
    torch.testing.assert_close(losses["reconstruction_loss"], torch.tensor(1.))
    torch.testing.assert_close(losses["kl_loss"], torch.tensor(1.5))
    torch.testing.assert_close(losses["loss"], expected_ce + 0.2 + 0.3 * 1.5)


def test_mu_classification_is_deterministic_and_state_roundtrip_matches():
    torch.manual_seed(12)
    config = ModelConfig(sequence_length=64)
    model = StateClassifier(config)
    x = signal()
    training_a, training_b = model(x), model(x)
    assert not torch.equal(training_a["z"], training_b["z"])
    torch.testing.assert_close(training_a["logits"], training_b["logits"], rtol=0, atol=0)
    model.eval()
    with torch.no_grad():
        expected = model(x)
        again = model(x)
    torch.testing.assert_close(expected["reconstruction"], again["reconstruction"], rtol=0, atol=0)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = StateClassifier(config).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.no_grad():
        actual = restored(x)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_weighted_reconstruction_requires_signal_target():
    model = StateClassifier(ModelConfig(sequence_length=64))
    output = model(signal())
    with pytest.raises(ValueError, match="x is required"):
        classification_loss(output, torch.tensor([0, 1, 2]))
