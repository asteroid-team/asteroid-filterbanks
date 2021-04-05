from torch import nn
import torch
from . import transforms


class ExponentialMovingAverage(nn.Module):
    def __init__(
        self,
        smooth: float = 0.04,
        per_channel: bool = False,
        n_channels: int = 2,
        trainable: bool = False,
    ):
        super().__init__()
        if per_channel:
            self._weights = nn.Parameter(
                torch.full((n_channels,), fill_value=smooth), requires_grad=trainable
            )
        else:
            self._weights = nn.Parameter(
                torch.full((1,), fill_value=smooth), requires_grad=trainable
            )

    def forward(self, mag_spec, initial_state):
        weights = torch.clamp(self._weights, 0.0, 1.0)
        accumulator = initial_state
        out = None
        for x in torch.split(mag_spec, 1, dim=-1):
            accumulator = weights * x + (1.0 - weights) * accumulator
            if out is None:
                out = accumulator
            else:
                out = torch.cat((out, accumulator), dim=-1)
        return out


class PCEN(nn.Module):
    """
    Per-Channel Energy Normalization

    Example implementations:

    https://github.com/google-research/leaf-audio/blob/master/leaf_audio/postprocessing.py#L142
    https://github.com/denfed/leaf-audio-pytorch/blob/main/leaf_audio_pytorch/postprocessing.py#L57
    https://github.com/f0k/ismir2018/blob/51067ca1b098307b8e12891445e9dd4aed96eab5/experiments/model.py

    [1] - https://arxiv.org/pdf/1607.05666.pdf
    """

    def __init__(
        self,
        alpha: float = 0.96,
        smooth: float = 0.04,
        delta: float = 2.0,
        root: float = 2.0,
        floor: float = 1e-6,
        trainable: bool = False,
        per_channel_smoothing: bool = False,
        n_channels: int = 2,
    ):
        super().__init__()

        self.floor = floor
        self.alpha = nn.Parameter(
            torch.full((n_channels,), fill_value=alpha), requires_grad=trainable
        )
        self.delta = nn.Parameter(
            torch.full((n_channels,), fill_value=delta), requires_grad=trainable
        )
        self.root = nn.Parameter(
            torch.full((n_channels,), fill_value=root), requires_grad=trainable
        )

        self.ema = ExponentialMovingAverage(
            smooth=smooth,
            per_channel=per_channel_smoothing,
            n_channels=n_channels,
            trainable=trainable,
        )

    def forward(self, spec: torch.Tensor):
        """
        [batch_size, n_channels, n_fft, timestep]
        """
        mag_spec = transforms.mag(spec, dim=-2)

        alpha = torch.min(self.alpha, torch.tensor(1.0))
        root = torch.max(self.root, torch.tensor(1.0))
        one_over_root = 1.0 / root

        initial_state = mag_spec[:, :, :, 0:1]
        ema_smoother = self.ema(mag_spec, initial_state=initial_state)
        out = (
            mag_spec.transpose(1, -1) / (self.floor + ema_smoother.transpose(1, -1)) ** alpha
            + self.delta
        ) ** one_over_root - self.delta ** one_over_root

        return out.transpose(1, -1)
