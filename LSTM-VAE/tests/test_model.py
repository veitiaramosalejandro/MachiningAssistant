"""Numerical and interface checks for the LSTM waveform VAE."""

import io
from dataclasses import asdict
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import ModelConfig, VAE, loss_function


@pytest.fixture(autouse=True)
def reproducible_cpu():
    torch.manual_seed(17)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous_threads)


def waveform(batch=2, length=64):
    t = torch.arange(length, dtype=torch.float32) / length
    pair = torch.stack((torch.sin(2 * torch.pi * 3 * t), 0.5 * torch.cos(2 * torch.pi * 9 * t)), dim=-1)
    return pair.unsqueeze(0).repeat(batch, 1, 1)


def small_model():
    return VAE(ModelConfig(sequence_length=64, hidden_size=12, latent_dim=4))


def test_default_2048_sample_forward():
    model = VAE(ModelConfig()).eval()
    with torch.no_grad():
        output = model(waveform(batch=1, length=2048))
    assert output['reconstruction'].shape == (1, 2048, 2)
    assert output['mu'].shape == output['logvar'].shape == output['z'].shape == (1, 16)
    assert all(torch.isfinite(value).all() for value in output.values())


def test_sampling_and_deterministic_inference():
    model = small_model()
    x = waveform()
    first, second = model(x), model(x)
    assert not torch.equal(first['z'], second['z'])
    assert not torch.equal(first['reconstruction'], second['reconstruction'])
    model.eval()
    first, second = model(x), model(x)
    torch.testing.assert_close(first['z'], first['mu'], rtol=0, atol=0)
    torch.testing.assert_close(first['reconstruction'], second['reconstruction'], rtol=0, atol=0)
    assert not torch.equal(model(x, sample=True)['z'], first['mu'])


def test_loss_has_known_gaussian_kl():
    target = torch.zeros(2, 3, 2)
    output = {'reconstruction': torch.ones_like(target), 'mu': torch.ones(2, 4), 'logvar': torch.zeros(2, 4)}
    losses = loss_function(output, target, beta=0.25)
    # Four independent N(1, 1) dimensions each contribute KL=0.5 to N(0, 1).
    torch.testing.assert_close(losses['reconstruction_loss'], torch.tensor(1.0))
    torch.testing.assert_close(losses['kl_loss'], torch.tensor(2.0))
    torch.testing.assert_close(losses['loss'], torch.tensor(1.5))


def test_reconstruction_and_kl_gradients_reach_encoder_and_decoder():
    model = small_model()
    losses = loss_function(model(waveform()), waveform())
    losses['reconstruction_loss'].backward()
    for parameter in [model.encoder.weight_ih_l0, model.to_mu.weight, model.to_logvar.weight,
                      model.decoder.weight_ih_l0, model.to_output.weight]:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    loss_function(model(waveform()), waveform())['kl_loss'].backward()
    for parameter in [model.encoder.weight_ih_l0, model.to_mu.weight, model.to_logvar.weight]:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert model.to_output.weight.grad is None


def test_optimizer_update_and_state_dict_round_trip():
    model = small_model()
    x = waveform()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    before = model.to_output.weight.detach().clone()
    optimizer.zero_grad()
    loss_function(model(x), x)['loss'].backward()
    optimizer.step()
    assert not torch.equal(before, model.to_output.weight)
    model.eval()
    expected = model(x)['reconstruction'].detach()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = small_model().eval()
    restored.load_state_dict(torch.load(buffer, map_location='cpu', weights_only=True))
    torch.testing.assert_close(restored(x)['reconstruction'], expected, rtol=0, atol=0)


def test_decoder_uses_the_latent_code():
    model = small_model().eval()
    z = torch.zeros(2, 4)
    z[1] = 1
    reconstructed = model.decode(z)
    assert not torch.allclose(reconstructed[0], reconstructed[1])


@pytest.mark.parametrize('shape', [(2, 64), (2, 63, 2), (2, 64, 1)])
def test_rejects_wrong_window_shape(shape):
    with pytest.raises(ValueError, match='Expected input shape'):
        small_model()(torch.zeros(shape))


@pytest.mark.parametrize('lower,upper', [(None, 10.), (-10., None),
    (float('nan'), 10.), (-10., float('inf')), (2., 2.), (3., 2.)])
def test_invalid_posterior_bounds_are_rejected(lower, upper):
    with pytest.raises(ValueError, match='logvar'):
        ModelConfig(sequence_length=64, logvar_min=lower, logvar_max=upper)


def test_extreme_raw_logvariance_is_finite_through_sampling_kl_and_backward():
    model = VAE(ModelConfig(sequence_length=64, hidden_size=12, latent_dim=4,
                            logvar_min=-10., logvar_max=10.))
    with torch.no_grad():
        model.to_logvar.weight.zero_()
        model.to_logvar.bias.copy_(torch.tensor([-100., 100., -.5, .5]))
    x = waveform()
    output = model(x, sample=True)
    torch.testing.assert_close(output['logvar'], torch.tensor([[-10., 10., -.5, .5]]).repeat(2, 1))
    assert all(torch.isfinite(value).all() for value in output.values())
    losses = loss_function(output, x)
    assert all(torch.isfinite(value).all() for value in losses.values())
    variance_gradient, = torch.autograd.grad(losses['kl_loss'], model.to_logvar.bias, retain_graph=True)
    assert variance_gradient[:2].count_nonzero() == 0
    assert variance_gradient[2:].abs().min() > 0
    losses['loss'].backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True))


@pytest.mark.parametrize('reconstruct', [False, True])
def test_classifier_inherits_encoder_bounds_in_both_paths(reconstruct):
    from classifier import StateClassifier, classification_loss
    model = StateClassifier(ModelConfig(sequence_length=64, hidden_size=12, latent_dim=4,
                                        logvar_min=-10., logvar_max=10.)).train()
    with torch.no_grad():
        model.vae.to_logvar.weight.zero_()
        model.vae.to_logvar.bias.copy_(torch.tensor([-100., 100., -.5, .5]))
    x = waveform()
    output = model(x, reconstruct=reconstruct)
    torch.testing.assert_close(output['logvar'], torch.tensor([[-10., 10., -.5, .5]]).repeat(2, 1))
    losses = classification_loss(output, torch.tensor([0, 2]), x=x if reconstruct else None,
                                 reconstruction_weight=.05 if reconstruct else 0.)
    assert all(torch.isfinite(value).all() for value in output.values())
    assert all(torch.isfinite(value).all() for value in losses.values())
    losses['loss'].backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_old_config_and_strict_state_loading_preserve_unbounded_default_behavior():
    legacy_config = {'sequence_length': 64, 'input_channels': 2, 'latent_dim': 4,
                     'hidden_size': 12, 'num_layers': 1}
    original = VAE(ModelConfig(**legacy_config)).eval()
    with torch.no_grad():
        original.to_logvar.weight.zero_()
        original.to_logvar.bias.copy_(torch.tensor([-100., 100., -.5, .5]))
    buffer = io.BytesIO()
    torch.save({'model_config': legacy_config, 'state_dict': original.state_dict()}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, map_location='cpu', weights_only=True)
    restored = VAE(ModelConfig(**checkpoint['model_config'])).eval()
    restored.load_state_dict(checkpoint['state_dict'], strict=True)
    assert restored.config.frame_size == 1 and restored.config.recurrent_steps == 64
    assert restored.config.logvar_min is None and restored.config.logvar_max is None
    expected = original(waveform(), sample=False)
    actual = restored(waveform(), sample=False)
    torch.testing.assert_close(actual['logvar'], torch.tensor([[-100., 100., -.5, .5]]).repeat(2, 1), rtol=0, atol=0)
    torch.testing.assert_close(actual['reconstruction'], expected['reconstruction'], rtol=0, atol=0)
    bounded_config = ModelConfig(**legacy_config, logvar_min=-10., logvar_max=10.)
    bounded = VAE(ModelConfig(**asdict(bounded_config))).eval()
    bounded.load_state_dict(checkpoint['state_dict'], strict=True)
    assert set(bounded.state_dict()) == set(original.state_dict())
    assert asdict(bounded.config)['logvar_min'] == -10. and asdict(bounded.config)['logvar_max'] == 10.
    torch.testing.assert_close(bounded(waveform(), sample=False)['reconstruction'], expected['reconstruction'], rtol=0, atol=0)


def test_legacy_frame_one_state_shapes_and_forward_are_unchanged():
    config = {'sequence_length': 64, 'input_channels': 2, 'latent_dim': 4,
              'hidden_size': 12, 'num_layers': 1}
    legacy = torch.nn.Module()
    legacy.encoder = torch.nn.LSTM(2, 12, num_layers=1, batch_first=True)
    legacy.to_mu = torch.nn.Linear(12, 4)
    legacy.to_logvar = torch.nn.Linear(12, 4)
    legacy.latent_to_hidden = torch.nn.Linear(4, 12)
    legacy.latent_to_cell = torch.nn.Linear(4, 12)
    legacy.decoder = torch.nn.LSTM(4, 12, num_layers=1, batch_first=True)
    legacy.to_output = torch.nn.Linear(12, 2)
    legacy.eval()
    restored = VAE(ModelConfig(**config)).eval()
    expected_shapes = {key: tuple(value.shape) for key, value in legacy.state_dict().items()}
    assert {key: tuple(value.shape) for key, value in restored.state_dict().items()} == expected_shapes
    restored.load_state_dict(legacy.state_dict(), strict=True)
    x = waveform()
    with torch.no_grad():
        _, (hidden, _) = legacy.encoder(x)
        mu = legacy.to_mu(hidden[-1])
        logvar = legacy.to_logvar(hidden[-1])
        h0 = torch.tanh(legacy.latent_to_hidden(mu)).reshape(2, 1, 12).transpose(0, 1).contiguous()
        c0 = torch.tanh(legacy.latent_to_cell(mu)).reshape(2, 1, 12).transpose(0, 1).contiguous()
        repeated = mu[:, None].expand(-1, 64, -1).contiguous()
        decoded, _ = legacy.decoder(repeated, (h0, c0))
        expected = legacy.to_output(decoded)
        actual = restored(x, sample=False)
    assert restored.config.frame_size == 1
    torch.testing.assert_close(actual['mu'], mu, rtol=0, atol=0)
    torch.testing.assert_close(actual['logvar'], logvar, rtol=0, atol=0)
    torch.testing.assert_close(actual['reconstruction'], expected, rtol=0, atol=0)


def test_complete_five_second_signal_uses_500_encoder_and_decoder_steps():
    config = ModelConfig(sequence_length=220500, frame_size=441)
    model = VAE(config).eval()
    x = waveform(batch=1, length=220500)
    captured = {}

    def encoder_input(module, args):
        captured['encoder'] = args[0].detach().clone()

    def decoder_input(module, args):
        captured['decoder'] = args[0].detach().clone()

    def frame_output(module, args, output):
        captured['output'] = output.detach().clone()

    handles = [model.encoder.register_forward_pre_hook(encoder_input),
               model.decoder.register_forward_pre_hook(decoder_input),
               model.to_output.register_forward_hook(frame_output)]
    try:
        with torch.no_grad():
            result = model(x, sample=False)
    finally:
        for handle in handles:
            handle.remove()
    assert config.recurrent_steps == 500
    assert captured['encoder'].shape == (1, 500, 882)
    assert captured['decoder'].shape == (1, 500, 16)
    assert captured['output'].shape == (1, 500, 882)
    assert result['reconstruction'].shape == (1, 220500, 2)
    torch.testing.assert_close(captured['encoder'].reshape_as(x), x, rtol=0, atol=0)
    torch.testing.assert_close(result['reconstruction'], captured['output'].reshape_as(x), rtol=0, atol=0)
    assert all(torch.isfinite(value).all() for value in result.values())


def test_nondivisible_framed_model_keeps_last_valid_sample_and_backward():
    model = VAE(ModelConfig(sequence_length=65, frame_size=8, hidden_size=12, latent_dim=4))
    x = waveform(length=65).requires_grad_()
    output = model(x)
    assert model.config.recurrent_steps == 9
    assert output['reconstruction'].shape == x.shape == (2, 65, 2)
    encoder_gradient, = torch.autograd.grad(
        output['mu'].square().sum()+output['logvar'].square().sum(), x, retain_graph=True,
    )
    assert torch.isfinite(encoder_gradient).all()
    assert encoder_gradient[:, -1].abs().sum() > 0
    loss_function(output, x)['loss'].backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[:, -1].abs().sum() > 0
    for parameter in [model.encoder.weight_ih_l0, model.to_output.weight]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
