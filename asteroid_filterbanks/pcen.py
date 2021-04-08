from torch import nn
import torch
from . import transforms
from typing import Union, Optional, Tuple

try:
    from typing import TypedDict
except ImportError:  # Fallback for >= Python 3.7
    from typing_extensions import TypedDict


class ExponentialMovingAverage(nn.Module):
    """
    Computes the exponential moving average of an sequential input.

    Influenced by leaf-audio's tensorflow implementation
        https://github.com/google-research/leaf-audio/blob/7ead2f9fe65da14c693c566fe8259ccaaf14129d/leaf_audio/postprocessing.py#L27

    See license here
        https://github.com/google-research/leaf-audio/blob/master/LICENSE
    """

    def __init__(
        self,
        smooth: float = 0.04,
        per_channel: bool = False,
        n_channels: int = 1,
        trainable: bool = False,
    ):
        super().__init__()
        if per_channel:
            weights = torch.full((n_channels,), fill_value=smooth)
        else:
            weights = torch.full((1,), fill_value=smooth)

        self.set_weights(data=weights, trainable=trainable)

    def set_weights(self, data: torch.Tensor, trainable: bool):
        self.weights = nn.Parameter(data, requires_grad=trainable)
        self.trainable = trainable

    def forward(
        self, mag_spec: torch.Tensor, initial_state: Optional[torch.Tensor]
    ) -> Tuple[torch.tensor, torch.Tensor]:
        if initial_state is None:
            initial_state = mag_spec[:, :, :, 0].unsqueeze(-1)

        weights = torch.clamp(self.weights, 0.0, 1.0)
        accumulator = initial_state.transpose(1, -1)
        out = []
        for x in torch.split(mag_spec.transpose(1, -1), 1, dim=1):
            accumulator = weights * x + (1.0 - weights) * accumulator
            out.append(accumulator.transpose(1, -1))
        return torch.cat(out, dim=-1), accumulator.transpose(1, -1)


class TrainableParameters(TypedDict):
    alpha: bool
    delta: bool
    root: bool
    smooth: bool


class _PCEN(nn.Module):
    def __init__(
        self,
        alpha: float = 0.96,
        delta: float = 2.0,
        root: float = 2.0,
        floor: float = 1e-6,
        smooth: float = 0.04,
        n_channels: int = 1,
        trainable: Union[bool, TrainableParameters] = False,
        per_channel_smoothing: bool = False,
    ):
        super().__init__()

        if trainable is True or trainable is False:
            trainable = TrainableParameters(
                alpha=trainable, delta=trainable, root=trainable, smooth=trainable
            )

        self.trainable = trainable
        self.n_channels = n_channels

        self.alpha = nn.Parameter(
            torch.full((self.n_channels,), fill_value=alpha), requires_grad=self.trainable["alpha"]
        )
        self.delta = nn.Parameter(
            torch.full((self.n_channels,), fill_value=delta), requires_grad=self.trainable["delta"]
        )
        self.root = nn.Parameter(
            torch.full((self.n_channels,), fill_value=root), requires_grad=self.trainable["root"]
        )

        self.floor = floor
        self.ema = ExponentialMovingAverage(
            smooth=smooth,
            per_channel=per_channel_smoothing,
            n_channels=self.n_channels,
            trainable=self.trainable["smooth"],
        )

    def forward(
        self, mag_spec: torch.Tensor, initial_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        post_squeeze = False
        if len(mag_spec.shape) == 3:
            # If n_channels is 1, add a single dimension to keep the shape consistent with multichannel shapes.
            mag_spec = mag_spec.unsqueeze(1)
            post_squeeze = True

        alpha = torch.min(self.alpha, torch.tensor(1.0))
        root = torch.max(self.root, torch.tensor(1.0))
        one_over_root = 1.0 / root
        ema_smoother, hidden_state = self.ema(mag_spec, initial_state)
        mag_spec = mag_spec.transpose(1, -1)
        ema_smoother = ema_smoother.transpose(1, -1)
        # Equation (1) in [1]
        out = (
            mag_spec / (self.floor + ema_smoother) ** alpha + self.delta
        ) ** one_over_root - self.delta ** one_over_root
        out = out.transpose(1, -1)
        if post_squeeze:
            out = out.squeeze(1)
        return out, hidden_state


class PCEN(_PCEN):
    """Per-Channel Energy Normalization as described in [1].

    This applies a fixed or learnable normalization by an exponential moving average smoother and a compression.

    Args:
        alpha: The AGC strength (or gain normalization strength) (α in the paper).
            Defaults to 0.96
        delta: Bias added before compression (δ in the paper).
            Defaults to 2.0
        root: One over exponent applied for compression (r in the paper).
            Defaults to 2.0
        floor: Offset added to compression, prevents division by zero. (ϵ in the paper)
            Defaults to 1e-6
        smooth: Smoothing coefficient (s in the paper).
            Defaults to 0.04
        n_channels: Number of channels in the time frequency representation.
            Defaults to 1
        trainable: If True, the parameters (alpha, delta, root and smooth) are trainable. If False, the parameters are fixed.
            Individual parameters can set to be fixed or trainable by passing a dictionary of booleans, with the key
            matching the parameter name and the value being either True (trainable) or False (fixed).
            i.e. ``{"alpha": False, "delta": True, "root": False, "smooth": True}``
            Defaults to False
        per_channel_smoothing: If True, each channel has it's own smoothing coefficient.
            Defaults to False

    Examples

        >>> audio = torch.randn(10, 2, 16_000)
        >>> fb = STFTFB(kernel_size=256, n_filters=256, stride=128)
        >>> enc = Encoder(fb)
        >>> tf_rep = enc(audio)
        >>> mag_spec = transform.mag(tf_rep)
        >>> pcen = PCEN(n_channels=2)
        >>> energy = pcen(mag_spec)
        >>> # or alternative with a mel-spectrogram
        >>> mel = MelScale(n_filters=256, sample_rate=16_000)
        >>> mel_spec = mel(audio)
        >>> energy = pcen(mag_spec)

    References
        [1]: Wang, Y., et al. "Trainable Frontend For Robust and Far-Field Keyword Spotting”, arXiv e-prints, 2016.
             https://arxiv.org/pdf/1607.05666.pdf

    Influenced by leaf-audio's tensorflow implementation
        https://github.com/google-research/leaf-audio/blob/7ead2f9fe65da14c693c566fe8259ccaaf14129d/leaf_audio/postprocessing.py

    See license here
        https://github.com/google-research/leaf-audio/blob/master/LICENSE
    """

    def forward(self, mag_spec: torch.Tensor) -> torch.Tensor:
        """Computes the PCEN from magnitude spectrum representation.

        Args:
            mag_spec: A tensor containing an magnitude spectrum representation.

        Shapes
            >>> (batch, n_channels, freq, time) -> (batch, n_channels, freq, time)
            >>> (batch, freq, time) -> (batch, freq, time) # Assumed to be mono-channel

        """
        return super().forward(mag_spec, None)[0]


class StatefulPCEN(_PCEN):
    """Per-Channel Energy Normalization as described in [1].

    Suitable for continuous real-time processing of incoming audio.

    This applies a fixed or learnable normalization by an exponential moving average smoother and a compression.

    Args:
        alpha: The AGC strength (or gain normalization strength) (α in the paper).
            Defaults to 0.96
        delta: Bias added before compression (δ in the paper).
            Defaults to 2.0
        root: One over exponent applied for compression (r in the paper).
            Defaults to 2.0
        floor: Offset added to compression, prevents division by zero. (ϵ in the paper)
            Defaults to 1e-6
        smooth: Smoothing coefficient (s in the paper).
            Defaults to 0.04
        n_channels: Number of channels in the time frequency representation.
            Defaults to 1
        trainable: If True, the parameters (alpha, delta, root and smooth) are trainable. If False, the parameters are fixed.
            Individual parameters can set to be fixed or trainable by passing a dictionary of booleans, with the key
            matching the parameter name and the value being either True (trainable) or False (fixed).
            i.e. ``{"alpha": False, "delta": True, "root": False, "smooth": True}``
            Defaults to False
        per_channel_smoothing: If True, each channel has it's own smoothing coefficient.
            Defaults to False

    Examples
        >>> audio = torch.randn(10, 2, 16_000)
        >>> fb = STFTFB(kernel_size=256, n_filters=256, stride=128)
        >>> enc = Encoder(fb)
        >>> tf_rep = enc(audio)
        >>> mag_spec = transform.mag(tf_rep)
        >>> mag_spec_1, mag_spec_2 = torch.chunk(mag_spec, 2, dim=-1)
        >>> pcen = PCEN(n_channels=2)
        >>> energy, hidden = pcen(mag_spec_1)
        >>> energy, hidden = pcen(mag_spec_2, hidden)

    References
        [1]: Wang, Y., et al. "Trainable Frontend For Robust and Far-Field Keyword Spotting”, arXiv e-prints, 2016.
             https://arxiv.org/pdf/1607.05666.pdf

    Influenced by leaf-audio's tensorflow implementation
        https://github.com/google-research/leaf-audio/blob/7ead2f9fe65da14c693c566fe8259ccaaf14129d/leaf_audio/postprocessing.py

    See license here
        https://github.com/google-research/leaf-audio/blob/master/LICENSE
    """

    @classmethod
    def from_pcen(cls, pcen: PCEN):
        stateful = cls()
        stateful.alpha = pcen.alpha
        stateful.delta = pcen.delta
        stateful.root = pcen.root
        stateful.floor = pcen.floor
        stateful.ema.set_weights(pcen.ema.weights.data, pcen.ema.trainable)
        return stateful

    def forward(
        self, mag_spec: torch.Tensor, initial_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the PCEN from magnitude spectrum representation, and an optional smoothed version of the filterbank (Equation (2) in [1]).

        Args:
            mag_spec: tensor containing an magnitude spectrum representation.
            initial_state: A tensor containing the initial hidden state.
                If set to None, defaults to the first element by time (dim=-1) in the mag_spec tensor.
                Defaults to None
        Shapes
            >>> (batch, n_channels, freq, time) -> (batch, n_channels, freq, time), (batch_size, n_channel, freq, 1)
            >>> (batch, freq, time) -> (batch, freq, time), (batch_size, 1, freq, 1) # Assumed to be mono-channel

        References
            [1]: Wang, Y., et al. "Trainable Frontend For Robust and Far-Field Keyword Spotting”, arXiv e-prints, 2016.
                https://arxiv.org/pdf/1607.05666.pdf
        """
        return super().forward(mag_spec, initial_state)
