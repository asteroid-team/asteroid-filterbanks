import torch
from asteroid_filterbanks import Encoder, Decoder, STFTFB
from asteroid_filterbanks.pcen import PCEN, ExponentialMovingAverage
from torch.testing import assert_allclose
import pytest
import re


@pytest.mark.parametrize("n_channels", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("n_filters", [128, 512])
@pytest.mark.parametrize("timesteps", [1, 10])
def test_ema(n_channels, batch_size, n_filters, timesteps):
    ema = ExponentialMovingAverage(n_channels=n_channels)

    mag_spec = torch.randn(batch_size, n_channels, n_filters, timesteps)
    initial_state = mag_spec[:, :, :, 0:1]

    out, hidden = ema(mag_spec, initial_state)
    assert out.shape == (batch_size, n_channels, n_filters, timesteps)
    assert hidden.shape == (batch_size, n_channels, n_filters, 1)


def test_pcen_init():
    pcen = PCEN(
        alpha=0.0,
        delta=0.5,
        root=1.0,
        floor=1.5,
        smooth=2.0,
        n_channels=2,
        per_channel_smoothing=True,
        trainable={"alpha": True, "delta": False, "root": True, "smooth": False},
    )

    assert_allclose(pcen.alpha.data, torch.tensor([0.0, 0.0]))
    assert pcen.alpha.requires_grad is True

    assert_allclose(pcen.delta.data, torch.tensor([0.5, 0.5]))
    assert pcen.delta.requires_grad is False

    assert_allclose(pcen.root.data, torch.tensor([1.0, 1.0]))
    assert pcen.root.requires_grad is True

    assert pcen.floor == 1.5

    assert_allclose(pcen.ema.weights.data, torch.tensor([2.0, 2.0]))
    assert pcen.ema.weights.requires_grad is False


@pytest.mark.parametrize("n_channels", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 10])
def test_pcen(n_channels, batch_size):
    audio = torch.randn(batch_size, n_channels, 16000 * 10)

    fb = STFTFB(kernel_size=256, n_filters=256, stride=128)
    enc = Encoder(fb)
    tf_rep = enc(audio)

    pcen = PCEN(n_channels=n_channels)
    energy, hidden = pcen(tf_rep)

    assert energy.shape == (batch_size, n_channels, 256 // 2 + 1, 1249)
    assert hidden.shape == (batch_size, n_channels, 256 // 2 + 1, 1)


@pytest.mark.parametrize("n_channels", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("n_filters", [128, 512])
@pytest.mark.parametrize("timesteps", [3, 10])
def test_pcen_hidden_state(n_channels, batch_size, n_filters, timesteps):
    pcen = PCEN(n_channels=n_channels)

    tf_rep = torch.randn((batch_size, n_channels, n_filters, timesteps))

    tf_rep_1, tf_rep_2 = torch.chunk(tf_rep, 2, dim=-1)

    energy_1, hidden = pcen(tf_rep_1)
    energy_2, hidden = pcen(tf_rep_2, hidden)
    energy = torch.cat((energy_1, energy_2), dim=-1)

    expected_energy, expected_hidden = pcen(tf_rep)
    assert_allclose(expected_energy, energy)
    assert_allclose(expected_hidden, hidden)


@pytest.mark.parametrize("n_channels", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("n_filters", [128, 512])
@pytest.mark.parametrize("timesteps", [1, 10])
def test_pcen_trainable(n_channels, batch_size, n_filters, timesteps):
    tf_rep = torch.randn((batch_size, n_channels, n_filters, timesteps))

    pcen = PCEN(trainable=True, n_channels=n_channels)
    energy_rep, _ = pcen(tf_rep)

    energy_rep.mean().backward()


def test_pcen_forward_shape_is_asteroid_complex():
    pcen = PCEN(trainable=True, n_channels=2)

    with pytest.raises(
        AssertionError,
        match=re.escape("Expected a complex-like tensor of shape (batch, n_channels, freq, time)."),
    ):
        pcen(torch.randn(10, 1, 1, 1))


@pytest.mark.parametrize("n_channels", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("n_filters", [128, 512])
@pytest.mark.parametrize("timesteps", [1, 10])
@pytest.mark.parametrize("trainable", [True, False])
def test_pcen_jit(n_channels, batch_size, n_filters, timesteps, trainable):
    tf_rep = torch.randn((batch_size, n_channels, n_filters, timesteps))
    pcen = PCEN(trainable=trainable, n_channels=n_channels)
    traced = torch.jit.trace(pcen, tf_rep)
    with torch.no_grad():
        native_out, native_hidden = pcen(tf_rep)
        jit_out, jit_hidden = traced(tf_rep)
        assert_allclose(native_out, jit_out)
        assert_allclose(native_hidden, jit_hidden)
