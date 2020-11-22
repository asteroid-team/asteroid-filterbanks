import warnings
from typing import List

import torch
import torch.nn.functional as F
from asteroid_filterbanks import Filterbank, STFTFB, Encoder, Decoder
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


def square_ola(window: torch.Tensor, kernel_size: int, stride: int, n_frame: int) -> torch.Tensor:
    window_sq = window.pow(2).view(1, -1, 1).repeat(1, 1, n_frame)
    return torch.nn.functional.fold(
        window_sq, (1, (n_frame - 1) * stride + kernel_size), (1, kernel_size), stride=(1, stride)
    ).squeeze(2)


@script_if_tracing
def pad_all_shapes(x: torch.Tensor, pad_shape: List[int], mode: str = "reflect") -> torch.Tensor:
    if x.ndim < 3:
        return F.pad(x[None, None], pad=pad_shape, mode=mode).squeeze(0).squeeze(0)
    return F.pad(x, pad=pad_shape, mode=mode)


class TorchSTFT(STFTFB):
    def __init__(self, *args, center=True, normalize=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.center = center
        self.normalize = normalize

    def pre_analysis(self, wav):
        if not self.center:
            return wav
        pad_shape = [self.kernel_size // 2, self.kernel_size // 2]
        wav = pad_all_shapes(wav, pad_shape=pad_shape, mode="reflect")
        return wav

    def post_analysis(self, spec):
        spec[..., 0, :] *= np.sqrt(2)
        spec[..., self.n_filters // 2, :] *= np.sqrt(2)
        return spec * (self.kernel_size // 2) ** 0.5

    def pre_synthesis(self, spec):
        spec = spec.clone()
        spec[..., 0, :] /= np.sqrt(2)
        spec[..., self.n_filters // 2, :] /= np.sqrt(2)
        return spec / (self.kernel_size // 2) ** 0.5

    def post_synthesis(self, wav):
        # Assume the window is the same in the STFT and ISTFT
        # OLA the squared windows, detect zeros and divide by it.
        # In pytorch this is the error.
        # RuntimeError: istft(torch.FloatTensor[129, 124, 2], n_fft=256,
        # hop_length=128, win_length=256, window=torch.FloatTensor{[256]},
        # center=0, normalized=0, onesided=1, length=16000)window overlap add min: 0

        n_frame = 1 + (wav.shape[-1] - self.kernel_size) // self.stride
        wsq_ola = square_ola(
            torch.from_numpy(self.torch_window),
            kernel_size=self.kernel_size,
            stride=self.stride,
            n_frame=n_frame,
        )
        view_shape = [1 for _ in wav.shape[:-1]] + [-1]
        wsq_ola = wsq_ola.view(view_shape)

        start = self.kernel_size // 2 if self.center else 0
        wav = wav[..., start:]
        wsq_ola = wsq_ola[..., start:]

        min_mask = wsq_ola.abs() < 1e-11
        if min_mask.any():
            # Warning instead of error. Might be trimmed afterward.
            warnings.warn(
                f"Minimum NOLA should be above 1e-11, Found {wsq_ola.abs().min()}. "
                f"Dividind only where possible."
            )
        wav[~min_mask] = wav[~min_mask] / wsq_ola[~min_mask]
        return wav


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
