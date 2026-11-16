"""Integration and gradient checks for the three-class LSTM-VAE wrapper."""

import io
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classifier import StateClassifier, classification_loss
from model import ModelConfig


@pytest.fixture(autouse=True)
def reproducible_cpu():
    torch.manual_seed(29)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous_threads)


def small_model():
    return StateClassifier(ModelConfig(sequence_length=64, hidden_size=12, latent_dim=4))


def batch():
    return torch.randn(3, 64, 2), torch.tensor([0, 1, 2])


def assert_nonzero_finite_grad(parameter):
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
    assert parameter.grad.abs().sum() > 0


def test_default_2048_window_has_three_logits_and_reconstruction():
    model = StateClassifier(ModelConfig()).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 2048, 2))
    assert output['logits'].shape == (1, 3)
    assert output['reconstruction'].shape == (1, 2048, 2)
    assert output['mu'].shape == output['logvar'].shape == output['z'].shape == (1, 16)
    assert all(torch.isfinite(value).all() for value in output.values())


def test_each_forward_encodes_once_and_classification_only_skips_decoder():
    model = small_model()
    x, _ = batch()
    calls = {'encoder': 0, 'decoder': 0}

    def encoder_hook(*_):
        calls['encoder'] += 1

    def decoder_hook(*_):
        calls['decoder'] += 1

    hooks = [model.vae.encoder.register_forward_hook(encoder_hook),
             model.vae.decoder.register_forward_hook(decoder_hook)]
    output = model(x, reconstruct=False)
    assert calls == {'encoder': 1, 'decoder': 0}
    assert 'reconstruction' not in output
    torch.testing.assert_close(output['z'], output['mu'], rtol=0, atol=0)
    model(x, reconstruct=True)
    assert calls == {'encoder': 2, 'decoder': 1}
    for hook in hooks:
        hook.remove()


def test_classification_is_based_on_mu_even_during_latent_sampling():
    model = small_model().train()
    x, _ = batch()
    first, second = model(x), model(x)
    assert not torch.equal(first['z'], second['z'])
    torch.testing.assert_close(first['logits'], second['logits'], rtol=0, atol=0)
    torch.testing.assert_close(first['logits'], model.head(first['mu']), rtol=0, atol=0)


def test_classification_gradients_reach_encoder_and_head():
    model = small_model()
    x, target = batch()
    loss = classification_loss(model(x, reconstruct=False), target)
    loss['ce_loss'].backward()
    for parameter in [model.vae.encoder.weight_ih_l0, model.vae.to_mu.weight,
                      model.head[0].weight, model.head[2].weight]:
        assert_nonzero_finite_grad(parameter)
    assert model.vae.to_logvar.weight.grad is None
    assert model.vae.decoder.weight_ih_l0.grad is None
    assert loss['reconstruction_loss'].item() == 0


def test_joint_objective_reaches_variational_and_decoder_parameters():
    model = small_model()
    x, target = batch()
    losses = classification_loss(model(x), target, x=x)
    losses['loss'].backward()
    for parameter in [model.vae.encoder.weight_ih_l0, model.vae.to_mu.weight,
                      model.vae.to_logvar.weight, model.vae.decoder.weight_ih_l0,
                      model.vae.to_output.weight, model.head[2].weight]:
        assert_nonzero_finite_grad(parameter)


def test_known_loss_scale_and_class_weight_support():
    x = torch.zeros(3, 5, 2)
    output = {'logits': torch.zeros(3, 3), 'reconstruction': torch.ones_like(x),
              'mu': torch.ones(3, 4), 'logvar': torch.zeros(3, 4)}
    losses = classification_loss(output, torch.tensor([0, 1, 2]), x=x,
                                 reconstruction_weight=0.05, beta=0.001,
                                 class_weights=torch.tensor([1., 2., 3.]))
    torch.testing.assert_close(losses['ce_loss'], torch.tensor(math.log(3)))
    torch.testing.assert_close(losses['reconstruction_loss'], torch.tensor(1.))
    torch.testing.assert_close(losses['kl_loss'], torch.tensor(2.))
    torch.testing.assert_close(losses['loss'], torch.tensor(math.log(3) + 0.05 + 0.002))


def test_eval_determinism_and_state_dict_reload():
    model = small_model().eval()
    x, _ = batch()
    expected = model(x)
    repeated = model(x)
    for key in expected:
        torch.testing.assert_close(expected[key], repeated[key], rtol=0, atol=0)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = small_model().eval()
    restored.load_state_dict(torch.load(buffer, map_location='cpu', weights_only=True))
    for key, value in restored(x).items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_reconstruction_target_is_required_when_loss_is_enabled():
    model = small_model()
    x, target = batch()
    with pytest.raises(ValueError, match='Provide x'):
        classification_loss(model(x), target)
