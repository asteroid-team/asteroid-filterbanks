import torch
import torch.nn as nn
import numpy as np
from .enc_dec import Filterbank

try:
    from torch import rfft, irfft

    def conj(filt):
        return torch.stack([filt[:, :, :, 1], -filt[:, :, :, 0]], dim=-1)


except ImportError:
    from torch import fft

    # Very bad "wrapper" of torch1.7 fft module to match torch.rfft and irfft
    # just for the analytic filterbanks. Signal_dim is ignore and signal_sizes
    # is assumed to be of length 1 if provided. Don't use this anywhere, this is
    # just a quick fix.
    def rfft(input, signal_ndim, normalized=False):
        norm = "ortho" if normalized else "backward"
        return fft.fft(input, dim=-1, norm=norm)

    def irfft(input, signal_ndim, normalized=False, signal_sizes=None):
        norm = "ortho" if normalized else "backward"
        n = None if signal_sizes is None else signal_sizes[0]
        return fft.irfft(input, n=n, dim=-1, norm=norm)

    def conj(filt):
        return filt.conj()


class AnalyticFreeFB(Filterbank):
    """Free analytic (fully learned with analycity constraints) filterbank.
    For more details, see [1].

    Args:
        n_filters (int): Number of filters. Half of `n_filters` will
            have parameters, the other half will be the hilbert transforms.
            `n_filters` should be even.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.
        sample_rate (float): Sample rate of the expected audio.
            Defaults to 8000.

    Attributes:
        n_feats_out (int): Number of output filters.

    References
        [1] : "Filterbank design for end-to-end speech separation". ICASSP 2020.
        Manuel Pariente, Samuele Cornell, Antoine Deleforge, Emmanuel Vincent.
    """

    def __init__(self, n_filters, kernel_size, stride=None, sample_rate=8000.0, **kwargs):
        super().__init__(n_filters, kernel_size, stride=stride, sample_rate=sample_rate)
        self.cutoff = int(n_filters // 2)
        self.n_feats_out = 2 * self.cutoff
        if n_filters % 2 != 0:
            print(
                "If the number of filters `n_filters` is odd, the "
                "output size of the layer will be `n_filters - 1`."
            )

        self._filters = nn.Parameter(torch.ones(n_filters // 2, 1, kernel_size), requires_grad=True)
        for p in self.parameters():
            nn.init.xavier_normal_(p, gain=1.0 / np.sqrt(2.0))

    def filters(self):
        ft_f = rfft(self._filters, 1, normalized=True)
        hft_f = conj(ft_f)
        hft_f = irfft(hft_f, 1, normalized=True, signal_sizes=(self.kernel_size,))
        return torch.cat([self._filters, hft_f], dim=0)
