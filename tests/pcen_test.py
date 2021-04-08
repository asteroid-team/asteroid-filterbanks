import torch
from asteroid_filterbanks import Encoder, Decoder, STFTFB, transforms
from asteroid_filterbanks.pcen import PCEN, StatefulPCEN, ExponentialMovingAverage
from torch.testing import assert_allclose
import pytest
import re


@pytest.mark.parametrize("n_channels", [2, 4])
@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("n_filters", [128, 512])
@pytest.mark.parametrize("timesteps", [1, 10])
@pytest.mark.parametrize("per_channel", [True, False])
@pytest.mark.parametrize("initial_state", [None, True])
def test_ema(n_channels, batch_size, n_filters, timesteps, per_channel, initial_state):
    ema = ExponentialMovingAverage(n_channels=n_channels, per_channel=True)
    mag_spec = torch.randn(batch_size, n_channels, n_filters, timesteps)
    if initial_state:
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
def test_pcen_forward(n_channels, batch_size):
    audio = torch.randn(batch_size, n_channels, 16000 * 10)

    fb = STFTFB(kernel_size=256, n_filters=256, stride=128)
    enc = Encoder(fb)
    tf_rep = enc(audio)
    mag_spec = transforms.mag(tf_rep)

    pcen = PCEN(n_channels=n_channels)
    energy = pcen(mag_spec)

    expected_shape = mag_spec.shape
    assert energy.shape == expected_shape


@pytest.mark.parametrize("n_channels", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("n_filters", [128, 512])
@pytest.mark.parametrize("timesteps", [1, 10])
def test_pcen_trainable(n_channels, batch_size, n_filters, timesteps):
    mag_spec = torch.randn((batch_size, n_channels, n_filters, timesteps))

    pcen = PCEN(trainable=True, n_channels=n_channels)
    energy_rep = pcen(mag_spec)

    energy_rep.mean().backward()


def test_stateful_pcen_init():
    pcen = StatefulPCEN(
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
@pytest.mark.parametrize("timesteps", [3, 10])
def test_stateful_pcen_forward(n_channels, batch_size, n_filters, timesteps):
    pcen = StatefulPCEN(n_channels=n_channels)

    mag_spec = torch.randn((batch_size, n_channels, n_filters, timesteps))
    mag_spec_1, mag_spec_2 = torch.chunk(mag_spec, 2, dim=-1)

    energy_1, hidden = pcen(mag_spec_1)
    energy_2, hidden = pcen(mag_spec_2, hidden)
    energy = torch.cat((energy_1, energy_2), dim=-1)

    expected_energy, expected_hidden = pcen(mag_spec)
    assert_allclose(expected_energy, energy)
    assert_allclose(expected_hidden, hidden)


@pytest.mark.parametrize("n_channels", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("n_filters", [128, 512])
@pytest.mark.parametrize("timesteps", [1, 10])
@pytest.mark.parametrize("trainable", [True, False])
def test_pcen_jit(n_channels, batch_size, n_filters, timesteps, trainable):
    mag_spec = torch.randn((batch_size, n_channels, n_filters, timesteps))
    pcen = PCEN(trainable=trainable, n_channels=n_channels)
    traced = torch.jit.trace(pcen, mag_spec)
    with torch.no_grad():
        native_energy = pcen(mag_spec)
        jit_energy = traced(mag_spec)
        assert_allclose(native_energy, jit_energy)


@pytest.mark.parametrize("n_channels", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("n_filters", [128, 512])
@pytest.mark.parametrize("timesteps", [1, 10])
@pytest.mark.parametrize("trainable", [True, False])
def test_stateful_pcen_jit(n_channels, batch_size, n_filters, timesteps, trainable):
    mag_spec = torch.randn((batch_size, n_channels, n_filters, timesteps))
    pcen = StatefulPCEN(trainable=trainable, n_channels=n_channels)
    traced = torch.jit.trace(pcen, mag_spec)
    with torch.no_grad():
        native_out, native_hidden = pcen(mag_spec)
        jit_out, jit_hidden = traced(mag_spec)
        assert_allclose(native_out, jit_out)
        assert_allclose(native_hidden, jit_hidden)


@pytest.mark.parametrize("n_channels", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("n_filters", [128, 512])
@pytest.mark.parametrize("timesteps", [3, 10])
@pytest.mark.parametrize("per_channel_smoothing", [True, False])
@pytest.mark.parametrize("trainable", [True, False])
def test_stateful_pcen_from_pcen(
    n_channels, batch_size, n_filters, timesteps, per_channel_smoothing, trainable
):
    mag_spec = torch.randn(batch_size, n_channels, n_filters, timesteps)

    pcen = PCEN(
        alpha=0.0,
        delta=0.5,
        root=1.0,
        floor=1.5,
        smooth=2.0,
        n_channels=n_channels,
        per_channel_smoothing=per_channel_smoothing,
        trainable=trainable,
    )

    stateful_pcen = StatefulPCEN.from_pcen(pcen)

    expected_energy = pcen(mag_spec)
    energy, hidden = stateful_pcen(mag_spec)

    assert_allclose(expected_energy, energy)
    assert hidden.shape == (batch_size, n_channels, n_filters, 1)
