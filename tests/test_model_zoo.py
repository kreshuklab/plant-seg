import torch

from plantseg.models.model import UNet2D
from plantseg.predictions.functional.utils import get_model_config

# test one 3d and one 3d model
MODEL_NAMES = ['confocal_2D_unet_ovules_ds2x', 'generic_confocal_3D_unet']


class TestModelZoo:
    def test_model_zoo(self):
        for model_name in MODEL_NAMES:
            model, _, model_path = get_model_config(model_name, model_update=True)
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
