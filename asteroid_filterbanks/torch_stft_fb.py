import warnings
from typing import Tuple
import torch
import torch.nn.functional as F
from asteroid_filterbanks import STFTFB
from asteroid_filterbanks.scripting import script_if_tracing


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

    .. note::
        When no `length` argument is given to `torch.istft`, the padding is
        aggressively removed. In this implementation, the whole signal is passed
        back to the user.
    """

    def __init__(
        self,
        n_filters,
        kernel_size,
        stride=None,
        window=None,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        sample_rate=8000.0,
        **kwargs,
    ):
        if n_filters != kernel_size:
            raise NotImplementedError(
                "Cannot set `n_filters!=kernel_size` in TorchSTFTFB, untested."
            )
        super().__init__(
            n_filters, kernel_size, stride=stride, window=window, sample_rate=sample_rate, **kwargs
        )
        self.center = center
        self.pad_mode = pad_mode
        if normalized:
            raise NotImplementedError
        if not onesided:
            raise NotImplementedError
        self.normalize = normalized

    @classmethod
    def from_torch_args(
        cls,
        n_fft,
        hop_length=None,
        win_length=None,
        window=None,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        sample_rate=8000.0,
        **kwargs,
    ):
        return cls(
            n_filters=n_fft,
            kernel_size=win_length,
            stride=hop_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            normalized=normalized,
            onesided=onesided,
            sample_rate=sample_rate,
            **kwargs,
        )

    def pre_analysis(self, wav):
        """Centers the frames if `center` is True."""
        if not self.center:
            return wav
        pad_shape = (self.kernel_size // 2, self.kernel_size // 2)
        wav = pad_all_shapes(wav, pad_shape=pad_shape, mode=self.pad_mode)
        return wav

    def post_analysis(self, spec):
        """Correct the scale to match torch.stft."""
        spec = _restore_freqs_an(spec, n_filters=self.n_filters)
        return spec * 0.5 * (self.kernel_size * self.n_filters / self.stride) ** 0.5

    def pre_synthesis(self, spec):
        """Correct the scale to match what's expected by torch.istft."""
        spec = spec.clone()
        spec = _restore_freqs_syn(spec, n_filters=self.n_filters)
        return spec / ((self.stride * self.n_filters / self.kernel_size) ** 0.5)

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
def _restore_freqs_an(spec, n_filters: int):
    spec[..., 0, :] *= 2 ** 0.5
    spec[..., n_filters // 2, :] *= 2 ** 0.5
    return spec


@script_if_tracing
def _restore_freqs_syn(spec, n_filters: int):
    spec[..., 0, :] /= 2 ** 0.5
    spec[..., n_filters // 2, :] /= 2 ** 0.5
    return spec


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
    wsq_ola = wsq_ola[..., start:].expand_as(wav)

    min_mask = wsq_ola.abs() < 1e-11
    if min_mask.any() and not torch.jit.is_scripting():
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
def pad_all_shapes(
    x: torch.Tensor, pad_shape: Tuple[int, int], mode: str = "reflect"
) -> torch.Tensor:
    if x.ndim == 1:
        return F.pad(x[None, None], pad=pad_shape, mode=mode).squeeze(0).squeeze(0)
    if x.ndim == 2:
        return F.pad(x[None], pad=pad_shape, mode=mode).squeeze(0)
    if x.ndim == 3:
        return F.pad(x, pad=pad_shape, mode=mode)
    pad_shape = (pad_shape[0],) + (0,) * (x.ndim - 1)
    return F.pad(x, pad=pad_shape, mode=mode)
