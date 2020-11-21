import torch
from asteroid_filterbanks import Filterbank, STFTFB, Encoder, Decoder
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


class TorchSTFT(STFTFB):
    def post_analysis(self, spec):
        spec[..., 0, :] *= np.sqrt(2)
        spec[..., self.n_filters // 2, :] *= np.sqrt(2)
        return spec * (self.kernel_size // 2) ** 0.5

    def pre_synthesis(self, spec):
        spec = spec.clone()
        spec[..., 0, :] /= np.sqrt(2)
        spec[..., self.n_filters // 2, :] /= np.sqrt(2)
        return spec / (self.kernel_size // 2) ** 0.5


if __name__ == "__main__":
    from torch.testing import assert_allclose

    wav = torch.randn(16000)
    for kernel_size in [16, 32, 64, 128, 256, 512]:
        stride = kernel_size // 2

        window = torch.ones(kernel_size) / 2 ** 0.5

        fb = TorchSTFT(
            n_filters=kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            window=window.data.numpy(),
        )
        # window = torch.from_numpy(fb.window)
        enc = Encoder(fb)
        dec = Decoder(fb)

        spec = torch.stft(
            wav.squeeze(),
            n_fft=kernel_size,
            hop_length=stride,
            win_length=kernel_size,
            center=False,
            window=window,
        )
        wav_back = torch.istft(
            spec,
            n_fft=kernel_size,
            hop_length=stride,
            win_length=kernel_size,
            window=window,
            center=False,
            length=wav.shape[0],
        )

        spec = to_asteroid(spec.float())
        asteroid_spec = enc(wav)
        asteroid_wavback = dec(asteroid_spec)

        assert_allclose(spec, asteroid_spec)
        assert_allclose(
            wav_back[kernel_size:-kernel_size], asteroid_wavback[kernel_size:-kernel_size]
        )

    # plt.plot(wav.numpy(), "r")
    # plt.plot(asteroid_wavback.numpy(), "b+")
    # plt.plot(wav_back.numpy(), "k+")
    # plt.show()
