import pytest
import torch

from plantseg.training.model import UNet2D
from plantseg.models.zoo import model_zoo

# test some modes (3D and 2D)
MODEL_NAMES = ['confocal_2D_unet_ovules_ds2x',
               'generic_confocal_3D_unet',
               'lightsheet_2D_unet_root_ds1x',
               'generic_light_sheet_3D_unet']


class TestModelZoo:
    @pytest.mark.skip("github workflows do not allow to download models for security reason")
    def test_model_zoo(self):
        for model_name in MODEL_NAMES:
            model, _, model_path = model_zoo.get_model_by_name(model_name, model_update=True)
            state = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state)
            model.eval()
            if isinstance(model, UNet2D):
                x = torch.randn(4, 1, 260, 260)
            else:
                x = torch.randn(4, 1, 80, 160, 160)
            y = model(x)
            # assert output normalized
            assert torch.all(0 <= y) and torch.all(y <= 1)
