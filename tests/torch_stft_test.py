import torch
from torch.testing import assert_allclose
import pytest
from scipy.signal import get_window
from asteroid_filterbanks.torch_stft_fb import TorchSTFTFB
from asteroid_filterbanks import Encoder, Decoder


def stft3d(x: torch.Tensor, *args, **kwargs):
    """Multichannel functional wrapper for torch.stft

    Args:
        x (Tensor): audio waveform of
            shape (nb_samples, nb_channels, nb_timesteps)
    Returns:
        STFT (Tensor): complex stft of
            shape (nb_samples, nb_channels, nb_bins, nb_frames, complex=2)
            last axis is stacked real and imaginary
    """

    shape = x.size()

    # pack batch
    x = x.view(-1, shape[-1])

    stft_f = torch.stft(x, *args, **kwargs)

    # unpack batch
    stft_f = stft_f.view(shape[:-1] + stft_f.shape[-3:])
    return stft_f


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def to_asteroid(x):
    return torch.cat([x[..., 0], x[..., 1]], dim=-2)


@pytest.mark.parametrize("n_fft_next_pow", [False])
@pytest.mark.parametrize("hop_ratio", [1, 2, 4])
@pytest.mark.parametrize(
    "win_length",
    [16, 32, 64, 128, 256, 512, 1024, 2048, 12, 30, 58, 122, 238, 498, 1018],
)
@pytest.mark.parametrize("window", [None, "hann", "blackman", "hamming"])
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("pad_mode", ["reflect", "constant"])
@pytest.mark.parametrize("normalized", [False])  # True unsupported
@pytest.mark.parametrize("sample_rate", [8000.0])  # No impact
@pytest.mark.parametrize("pass_length", [True])
@pytest.mark.parametrize("wav_shape", [(8000,), (2, 3, 8000)])
def test_torch_stft(
    n_fft_next_pow,
    hop_ratio,
    win_length,
    window,
    center,
    pad_mode,
    normalized,
    sample_rate,
    pass_length,
    wav_shape,
):
    # Accept 0.1 less tolerance for larger windows.
    RTOL = 1e-3 if win_length > 256 else 1e-4
    ATOL = 1e-4 if win_length > 256 else 1e-5
    wav = torch.randn(wav_shape, dtype=torch.float32)
    output_len = wav.shape[-1] if pass_length else None
    n_fft = win_length if not n_fft_next_pow else next_power_of_2(win_length)
    hop_length = win_length // hop_ratio

    window = None if window is None else get_window(window, win_length, fftbins=True)
    if window is not None:
        # Cannot restore the signal without overlap and near to zero window.
        if hop_ratio == 1 and (window ** 2 < 1e-11).any():
            pass

    fb = TorchSTFTFB.from_torch_args(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=normalized,
        onesided=True,
        sample_rate=sample_rate,
    )

    stft = Encoder(fb)
    istft = Decoder(fb)

    spec = stft3d(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=fb.torch_window,
        center=center,
        pad_mode=pad_mode,
        normalized=normalized,
        onesided=True,
    )

    spec_asteroid = stft(wav)
    torch_spec = to_asteroid(spec.float())
    assert_allclose(spec_asteroid, torch_spec, rtol=RTOL, atol=ATOL)

    try:
        wav_back = torch.istft(
            spec,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=fb.torch_window,
            center=center,
            normalized=normalized,
            onesided=True,
            length=output_len,
        )

    except RuntimeError:
        # If there was a RuntimeError, the OLA had zeros. So we cannot unit test
        # But we can make sure that istft raises a warning about it.
        with pytest.warns(RuntimeWarning):
            _ = istft(spec_asteroid, length=output_len)
    else:
        # If there was no RuntimeError, we unit-test against the results.
        wav_back_asteroid = istft(spec_asteroid, length=output_len)
        # Asteroid always returns a longer signal.
        assert wav_back_asteroid.shape[-1] >= wav_back.shape[-1]
        # The unit test is done on the left part of the signal.
        assert_allclose(
            wav_back_asteroid[: wav_back.shape[-1]], wav_back.float(), rtol=RTOL, atol=ATOL
        )


def test_raises_if_onesided_is_false():
    with pytest.raises(NotImplementedError):
        TorchSTFTFB.from_torch_args(512, hop_length=256, win_length=256, onesided=False)


def test_raises_if_normalized_is_true():
    with pytest.raises(NotImplementedError):
        TorchSTFTFB.from_torch_args(512, hop_length=256, win_length=256, normalized=True)


def test_raises_filt_kern_diff():
    with pytest.raises(NotImplementedError):
        TorchSTFTFB.from_torch_args(512, win_length=500)
    with pytest.raises(NotImplementedError):
        TorchSTFTFB(n_filters=512, kernel_size=500)
