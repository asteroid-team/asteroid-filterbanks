import torch
from asteroid_filterbanks import Encoder, Decoder, STFTFB
from asteroid_filterbanks.pcen import PCEN, ExponentialMovingAverage


def test_ema():
    batch_size = 2
    n_channels = 2
    n_filters = 512
    timesteps = 10

    ema = ExponentialMovingAverage(n_channels=n_channels)

    mag_spec = torch.randn(batch_size, n_channels, n_filters, timesteps)
    initial_state = mag_spec[:, :, :, 0:1]

    y = ema(mag_spec, initial_state)
    assert y.shape == (batch_size, n_channels, n_filters, timesteps)


def test_pcen():
    fb = STFTFB(kernel_size=256, n_filters=256, stride=128)
    enc = Encoder(fb)

    batch_size = 2
    n_channels = 2

    audio = torch.randn(batch_size, n_channels, 16000 * 10)
    stft = enc(audio)

    pcen = PCEN(n_channels=2)
    E = pcen(stft)
    assert E.shape == (batch_size, n_channels, 256 // 2 + 1, 1249)
