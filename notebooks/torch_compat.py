import warnings
from typing import List

import torch
import torch.nn.functional as F
from asteroid_filterbanks import Filterbank, STFTFB, Encoder, Decoder
from asteroid_filterbanks.torch_stft_fb import TorchSTFTFB
from asteroid_filterbanks.scripting import script_if_tracing
from asteroid_filterbanks.transforms import mag
import matplotlib.pyplot as plt
import numpy as np


def all_mag(x, EPS=1e-7):
    if x.shape[-1] == 2:
        power = x.pow(2).sum(-1) + EPS
        return power.pow(0.5)
    return mag(x, -2, EPS=EPS)


def plot(*specs):
    for s in specs:
        mag_spec = all_mag(s).data.numpy()
        plt.figure()
        plt.imshow(mag_spec[0].T)
        plt.colorbar()
        plt.show(block=False)


def to_asteroid(x):
    return torch.cat([x[..., 0], x[..., 1]], dim=-2)


if __name__ == "__main__":
    from torch.testing import assert_allclose

    wav = torch.randn(16000)
    center = True
    for kernel_size in [16, 32, 64, 128, 256, 512]:
        print(kernel_size)
        stride = kernel_size // 2

        # window = torch.ones(kernel_size) / 2 ** 0.5
        window = 0.001 * torch.hann_window(kernel_size) ** 0.85

        fb = TorchSTFT(
            n_filters=kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            window=window.data.numpy(),
            center=center,
        )
        # window = torch.from_numpy(fb.window)
        enc = Encoder(fb)
        dec = Decoder(fb)

        spec = torch.stft(
            wav.squeeze(),
            n_fft=kernel_size,
            hop_length=stride,
            win_length=kernel_size,
            center=center,
            window=window,
        )
        wav_back = torch.istft(
            spec,
            n_fft=kernel_size,
            hop_length=stride,
            win_length=kernel_size,
            window=window,
            center=center,
            length=wav.shape[0],
        )
        spec = to_asteroid(spec.float())
        asteroid_spec = enc(wav)
        asteroid_wavback = dec(asteroid_spec, length=wav.shape[0])
        #
        assert_allclose(spec, asteroid_spec)
        assert_allclose(wav_back, asteroid_wavback)

    # plt.plot(wav.numpy(), "r")
    # plt.plot(asteroid_wavback.numpy(), "b+")
    # plt.plot(wav_back.numpy(), "k+")
    # plt.show()
