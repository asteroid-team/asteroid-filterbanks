import torch
import pytest
from torch.testing import assert_allclose
from asteroid_filterbanks import make_enc_dec
from asteroid_filterbanks.torch_stft_fb import TorchSTFTFB


@pytest.mark.parametrize(
    "filter_bank_name",
    ("free", "stft", "analytic_free", "param_sinc", TorchSTFTFB),
)
@pytest.mark.parametrize(
    "inference_data",
    (
        (torch.rand(240) - 0.5) * 2,
        (torch.rand(1, 220) - 0.5) * 2,
        (torch.rand(4, 256) - 0.5) * 2,
        (torch.rand(1, 3, 312) - 0.5) * 2,
        (torch.rand(3, 2, 128) - 0.5) * 2,
        (torch.rand(1, 1, 3, 212) - 0.5) * 2,
        (torch.rand(2, 4, 3, 128) - 0.5) * 2,
    ),
)
def test_jit_filterbanks_enc(filter_bank_name, inference_data):
    n_filters = 32
    if filter_bank_name == TorchSTFTFB:
        kernel_size = n_filters
    else:
        kernel_size = 2 * n_filters
    enc, _ = make_enc_dec(filter_bank_name, n_filters=n_filters, kernel_size=kernel_size)

    inputs = ((torch.rand(1, 200) - 0.5) * 2,)
    traced = torch.jit.trace(enc, inputs)
    with torch.no_grad():
        res = enc(inference_data)
        out = traced(inference_data)
        print(traced.code_with_constants)
        print(traced.code)
        assert_allclose(res, out)


@pytest.mark.parametrize(
    "filter_bank_name",
    ("free", "stft", "analytic_free", "param_sinc", TorchSTFTFB),
)
@pytest.mark.parametrize(
    "inference_data",
    (
        (torch.rand(240) - 0.5) * 2,
        (torch.rand(1, 220) - 0.5) * 2,
        (torch.rand(4, 256) - 0.5) * 2,
        (torch.rand(1, 3, 312) - 0.5) * 2,
        (torch.rand(3, 2, 128) - 0.5) * 2,
        (torch.rand(1, 1, 3, 212) - 0.5) * 2,
        (torch.rand(2, 4, 3, 128) - 0.5) * 2,
    ),
)
def test_jit_filterbanks(filter_bank_name, inference_data):
    model = DummyModel(fb_name=filter_bank_name)
    model = model.eval()

    inputs = ((torch.rand(1, 200) - 0.5) * 2,)
    traced = torch.jit.trace(model, inputs)
    with torch.no_grad():
        res = model(inference_data)
        out = traced(inference_data)
        print(traced.code_with_constants)
        print(traced.code)
        assert_allclose(res, out)


class DummyModel(torch.nn.Module):
    def __init__(
        self,
        fb_name="free",
        kernel_size=16,
        n_filters=32,
        stride=8,
        **fb_kwargs,
    ):
        super().__init__()
        if fb_name == TorchSTFTFB:
            n_filters = kernel_size
        encoder, decoder = make_enc_dec(
            fb_name, kernel_size=kernel_size, n_filters=n_filters, stride=stride, **fb_kwargs
        )
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, wav):
        tf_rep = self.encoder(wav)
        wav_back = self.decoder(tf_rep)
        return wav_back
