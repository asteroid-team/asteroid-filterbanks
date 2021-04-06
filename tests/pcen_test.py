import torch
from asteroid_filterbanks import Encoder, Decoder, STFTFB
from asteroid_filterbanks.pcen import PCEN, ExponentialMovingAverage
from torch.testing import assert_allclose
import pytest


@pytest.mark.parametrize("n_channels", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("n_filters", [128, 512])
@pytest.mark.parametrize("timesteps", [1, 10])
def test_ema(n_channels, batch_size, n_filters, timesteps):
    ema = ExponentialMovingAverage(n_channels=n_channels)

    mag_spec = torch.randn(batch_size, n_channels, n_filters, timesteps)
    initial_state = mag_spec[:, :, :, 0:1]

    out = ema(mag_spec, initial_state)
    assert out.shape == (batch_size, n_channels, n_filters, timesteps)


@pytest.mark.parametrize("n_channels", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 10])
def test_pcen(n_channels, batch_size):
    audio = torch.randn(batch_size, n_channels, 16000 * 10)

    fb = STFTFB(kernel_size=256, n_filters=256, stride=128)
    enc = Encoder(fb)
    tf_rep = enc(audio)

    pcen = PCEN(n_channels=n_channels)
    energy = pcen(tf_rep)

    assert energy.shape == (batch_size, n_channels, 256 // 2 + 1, 1249)


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
@pytest.mark.parametrize("n_filters", [128, 512])
@pytest.mark.parametrize("timesteps", [1, 10])
def test_pcen_trainable(n_channels, batch_size, n_filters, timesteps):
    tf_rep = torch.randn((batch_size, n_channels, n_filters, timesteps))

    pcen = PCEN(trainable=True, n_channels=n_channels)
    energy_rep = pcen(tf_rep)

    energy_rep.mean().backward()