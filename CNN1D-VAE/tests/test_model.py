"""Shape, variational behavior, optimization, and state persistence tests."""
import io
from dataclasses import asdict
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import ModelConfig, VAE, loss_function


torch.set_num_threads(1)


def signal(batch=2, length=64):
    t = torch.arange(length, dtype=torch.float32) / length
    x = torch.stack((torch.sin(2 * torch.pi * 5 * t), 0.4 * torch.cos(2 * torch.pi * 11 * t)), dim=-1)
    return x.unsqueeze(0).repeat(batch, 1, 1)


def test_default_window_and_batch_shapes_are_finite():
    model = VAE(ModelConfig()).eval()
    with torch.no_grad():
        out = model(signal(batch=3, length=2048))
    assert out["reconstruction"].shape == (3, 2048, 2)
    for name in ["mu", "logvar", "z"]:
        assert out[name].shape == (3, 16)
    assert all(torch.isfinite(value).all() for value in out.values())


def test_sampling_is_stochastic_only_when_requested():
    model = VAE(ModelConfig(sequence_length=64))
    x = signal()
    model.train()
    train_a, train_b = model(x), model(x)
    torch.testing.assert_close(train_a["mu"], train_b["mu"])
    assert not torch.equal(train_a["z"], train_b["z"])
    model.eval()
    a, b = model(x), model(x)
    torch.testing.assert_close(a["reconstruction"], b["reconstruction"], rtol=0, atol=0)
    torch.testing.assert_close(a["z"], a["mu"], rtol=0, atol=0)
    assert not torch.equal(model(x, sample=True)["z"], a["z"])
    model.train()
    torch.testing.assert_close(model(x, sample=False)["reconstruction"], a["reconstruction"], rtol=0, atol=0)


def test_loss_matches_known_gaussian_and_reconstruction_case():
    target = torch.zeros(2, 64, 2)
    out = {"reconstruction": torch.ones_like(target),
           "mu": torch.ones(2, 3), "logvar": torch.zeros(2, 3)}
    losses = loss_function(out, target, beta=0.2)
    torch.testing.assert_close(losses["reconstruction_loss"], torch.tensor(1.0))
    torch.testing.assert_close(losses["kl_loss"], torch.tensor(1.5))
    torch.testing.assert_close(losses["loss"], torch.tensor(1.3))


def test_reconstruction_and_kl_both_reach_encoder_heads():
    torch.manual_seed(7)
    model = VAE(ModelConfig(sequence_length=64))
    x = signal()
    output = model(x, sample=True)
    losses = loss_function(output, x)
    head_params = [model.mu_head.weight, model.logvar_head.weight]
    for loss_name in ["reconstruction_loss", "kl_loss"]:
        gradients = torch.autograd.grad(losses[loss_name], head_params, retain_graph=True)
        assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    losses["loss"].backward()
    assert model.decoder[-1].weight.grad.abs().sum() > 0


def test_optimizer_updates_and_can_reduce_reconstruction_loss():
    torch.manual_seed(11)
    model = VAE(ModelConfig(sequence_length=64))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    x = signal()
    before = loss_function(model(x, sample=False), x)["reconstruction_loss"].item()
    original_weight = model.encoder[0].weight.detach().clone()
    for _ in range(12):
        optimizer.zero_grad()
        losses = loss_function(model(x, sample=False), x)
        losses["loss"].backward()
        optimizer.step()
    after = loss_function(model(x, sample=False), x)["reconstruction_loss"].item()
    assert not torch.equal(original_weight, model.encoder[0].weight)
    assert after < before


def test_state_dict_round_trip_preserves_inference():
    torch.manual_seed(19)
    config = ModelConfig(sequence_length=64)
    model = VAE(config).eval()
    x = signal()
    with torch.no_grad():
        expected = model(x)["reconstruction"]
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = VAE(config).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.no_grad():
        actual = restored(x)["reconstruction"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_invalid_window_layout_and_dtype_fail_explicitly():
    with pytest.raises(ValueError, match="multiple of 64"):
        ModelConfig(sequence_length=100)
    model = VAE(ModelConfig(sequence_length=64))
    with pytest.raises(ValueError, match="Expected input shape"):
        model(torch.zeros(2, 2, 64))
    with pytest.raises(TypeError, match="float32"):
        model(torch.zeros(2, 64, 2, dtype=torch.int16))


@pytest.mark.parametrize('lower,upper', [(None, 10.), (-10., None),
    (float('nan'), 10.), (-10., float('inf')), (2., 2.), (3., 2.)])
def test_invalid_posterior_bounds_are_rejected(lower, upper):
    with pytest.raises(ValueError, match='logvar'):
        ModelConfig(sequence_length=64, logvar_min=lower, logvar_max=upper)


def test_extreme_raw_logvariance_is_finite_through_sampling_kl_and_backward():
    torch.manual_seed(23)
    model = VAE(ModelConfig(sequence_length=64, latent_dim=4, logvar_min=-10., logvar_max=10.))
    with torch.no_grad():
        model.logvar_head.weight.zero_()
        model.logvar_head.bias.copy_(torch.tensor([-100., 100., -.5, .5]))
    x = signal()
    output = model(x, sample=True)
    torch.testing.assert_close(output['logvar'], torch.tensor([[-10., 10., -.5, .5]]).repeat(2, 1))
    assert all(torch.isfinite(value).all() for value in output.values())
    losses = loss_function(output, x)
    assert all(torch.isfinite(value).all() for value in losses.values())
    variance_gradient, = torch.autograd.grad(losses['kl_loss'], model.logvar_head.bias, retain_graph=True)
    assert variance_gradient[:2].count_nonzero() == 0
    assert variance_gradient[2:].abs().min() > 0
    losses['loss'].backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True))


@pytest.mark.parametrize('reconstruct', [False, True])
def test_classifier_uses_the_same_bound_with_or_without_reconstruction(reconstruct):
    from classifier import StateClassifier, classification_loss
    model = StateClassifier(ModelConfig(sequence_length=64, latent_dim=4, logvar_min=-10., logvar_max=10.)).train()
    with torch.no_grad():
        model.vae.logvar_head.weight.zero_()
        model.vae.logvar_head.bias.copy_(torch.tensor([-100., 100., -.5, .5]))
    x = signal()
    output = model(x, reconstruct=reconstruct)
    torch.testing.assert_close(output['logvar'], torch.tensor([[-10., 10., -.5, .5]]).repeat(2, 1))
    losses = classification_loss(output, torch.tensor([0, 2]), x=x if reconstruct else None,
                                 reconstruction_weight=.05 if reconstruct else 0.)
    assert all(torch.isfinite(value).all() for value in output.values())
    assert all(torch.isfinite(value).all() for value in losses.values())
    losses['loss'].backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_old_config_and_strict_state_loading_preserve_unbounded_default_behavior():
    legacy_config = {'sequence_length': 64, 'input_channels': 2, 'latent_dim': 4}
    original = VAE(ModelConfig(**legacy_config)).eval()
    with torch.no_grad():
        original.logvar_head.weight.zero_()
        original.logvar_head.bias.copy_(torch.tensor([-100., 100., -.5, .5]))
    buffer = io.BytesIO()
    torch.save({'model_config': legacy_config, 'state_dict': original.state_dict()}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    restored = VAE(ModelConfig(**checkpoint['model_config'])).eval()
    restored.load_state_dict(checkpoint['state_dict'], strict=True)
    assert restored.config.logvar_min is None and restored.config.logvar_max is None
    expected = original(signal(), sample=False)
    actual = restored(signal(), sample=False)
    torch.testing.assert_close(actual['logvar'], torch.tensor([[-100., 100., -.5, .5]]).repeat(2, 1), rtol=0, atol=0)
    torch.testing.assert_close(actual['reconstruction'], expected['reconstruction'], rtol=0, atol=0)
    bounded_config = ModelConfig(**legacy_config, logvar_min=-10., logvar_max=10.)
    bounded = VAE(ModelConfig(**asdict(bounded_config))).eval()
    bounded.load_state_dict(checkpoint['state_dict'], strict=True)
    assert set(bounded.state_dict()) == set(original.state_dict())
    assert asdict(bounded.config)['logvar_min'] == -10. and asdict(bounded.config)['logvar_max'] == 10.
    torch.testing.assert_close(bounded(signal(), sample=False)['reconstruction'], expected['reconstruction'], rtol=0, atol=0)
