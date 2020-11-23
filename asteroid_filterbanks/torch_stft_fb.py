import warnings
from typing import List
import torch
import torch.nn.functional as F
from asteroid_filterbanks import STFTFB, Encoder, Decoder
from asteroid_filterbanks.scripting import script_if_tracing
import numpy as np


class TorchSTFTFB(STFTFB):
    """Equivalent to :function:`torch.stft` and :function:`torch.istft` using
    `asteroid-filterbanks`.

    Args:
        *args: Passed to `STFTFB`.
        center (bool): Whether to center the each frame. (pad left and right).
        pad_mode (str): How to pad if `center` is True. Only used for the STFT.
        normalize: Unsupported yet.
        **kwargs: Passed to `STFTFB`.

    See :class:`STFTFB`, :function:`torch.stft` and :function:`torch.istft` for
    more detail.

    .. warning::
        In order to obtain perfect reconstruction, `torch.istft` divides the wav
        output by the overlap-added square window (WSOLA) (assumed the same for
         analysis and synthesis). This might cause problems for windows containing
         zeros (and several common windows do). When using `center=True`, the leading
        and trailing zeros of the WSOLA will be discarded, virtually solving this
        problem.

        We intentionally match this behavior, to provide code that behaves exactly
        like `torch.istft`. But whereas `torch.istft` raises an error when the
        WSOLA contains zeros, we only raise a warning.

        As the code is rigorously tested, if your code worked before with
        `torch.istft`, it won't break with TorchSTFTFB.
    """

    def __init__(self, *args, center=True, pad_mode="reflect", normalize=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.center = center
        self.pad_mode = pad_mode
        if normalize:
            raise NotImplementedError
        self.normalize = normalize

    def pre_analysis(self, wav):
        """Centers the frames if `center` is True."""
        if not self.center:
            return wav
        pad_shape = [self.kernel_size // 2, self.kernel_size // 2]
        wav = pad_all_shapes(wav, pad_shape=pad_shape, mode=self.pad_mode)
        return wav

    def post_analysis(self, spec):
        """Correct the scale to match torch.stft."""
        spec[..., 0, :] *= np.sqrt(2)
        spec[..., self.n_filters // 2, :] *= np.sqrt(2)
        return spec * (self.kernel_size // 2) ** 0.5

    def pre_synthesis(self, spec):
        """Correct the scale to match what's expected by torch.istft."""
        spec = spec.clone()
        spec[..., 0, :] /= np.sqrt(2)
        spec[..., self.n_filters // 2, :] /= np.sqrt(2)
        return spec / (self.kernel_size // 2) ** 0.5

    def post_synthesis(self, wav):
        """Perform OLA on the waveforms and divide by the squared window OLA.
        Assumes that the window is the same in the STFT and iSTFT.
        """
        wav_out = ola_with_wdiv(
            wav,
            window=self.torch_window,
            kernel_size=self.kernel_size,
            stride=self.stride,
            center=self.center,
        )
        return wav_out


@script_if_tracing
def ola_with_wdiv(wav, window, kernel_size: int, stride: int, center: bool = True) -> torch.Tensor:
    n_frame = 1 + (wav.shape[-1] - kernel_size) // stride
    wsq_ola = square_ola(
        window,
        kernel_size=kernel_size,
        stride=stride,
        n_frame=n_frame,
    )
    view_shape = [1 for _ in wav.shape[:-1]] + [-1]
    wsq_ola = wsq_ola.view(view_shape)

    start = kernel_size // 2 if center else 0
    wav = wav[..., start:]
    wsq_ola = wsq_ola[..., start:]

    min_mask = wsq_ola.abs() < 1e-11
    if min_mask.any():
        # Warning instead of error. Might be trimmed afterward.
        warnings.warn(
            f"Minimum NOLA should be above 1e-11, Found {wsq_ola.abs().min()}. "
            f"Dividind only where possible.",
            RuntimeWarning,
        )
    wav[~min_mask] = wav[~min_mask] / wsq_ola[~min_mask]
    return wav


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
