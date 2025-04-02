import pytest
import torch

from plantseg.training.embeddings import embeddings_to_affinities
from plantseg.training.model import SpocoNet, UNet2D
from tests.conftest import IS_CUDA_AVAILABLE


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Cuda is not available")
class TestModelPrediction:
    def test_UNet2D(self):
        model = UNet2D(in_channels=3, out_channels=1)
        model.eval()
        x = torch.randn(4, 3, 260, 260)
        y = model(x)
        assert y.shape == (4, 1, 260, 260)
        assert torch.all(y >= 0) and torch.all(y <= 1)

    def test_SpocoNet(self):
        model = SpocoNet.from_unet_params(
            in_channels=1, out_channels=8, f_maps=[16, 32, 64, 128, 256]
        )
        model.eval()
        x1 = torch.randn(4, 1, 260, 260)
        x2 = torch.rand(4, 1, 260, 260)
        y1, y2 = model(x1, x2)
        assert y1.shape == (4, 8, 260, 260)
        assert y2.shape == (4, 8, 260, 260)

    def test_embeddings_to_affinities(self):
        x = torch.randn(4, 8, 128, 128)
        offsets = [[-1, 0], [0, -1]]
        delta = 0.5
        affs = embeddings_to_affinities(x, offsets, delta)
        assert affs.shape == (4, 2, 128, 128)
        assert torch.all(affs >= 0) and torch.all(affs <= 1)
