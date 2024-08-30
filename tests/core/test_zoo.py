import os

import pytest
import torch

from plantseg.core.zoo import model_zoo
from plantseg.training.model import UNet2D
from tests.conftest import IS_CUDA_AVAILABLE

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

# test some modes (3D and 2D)
MODEL_NAMES = [
    'confocal_2D_unet_ovules_ds2x',
    'generic_confocal_3D_unet',
    'lightsheet_2D_unet_root_ds1x',
    'generic_light_sheet_3D_unet',
]


class TestPlantSegModelZoo:
    """Test the PlantSeg model zoo"""

    @pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Cuda is not available")
    @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Github workflows do not allow model download for security reason")
    def test_model_output_normalisation(self):
        for model_name in MODEL_NAMES:
            model, _, model_path = model_zoo.get_model_by_name(model_name, model_update=True)
            state = torch.load(model_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state)
            model.eval()
            if isinstance(model, UNet2D):
                x = torch.randn(4, 1, 260, 260)
            else:
                x = torch.randn(4, 1, 80, 160, 160)
            y = model(x)
            # assert output normalized
            assert torch.all(0 <= y) and torch.all(y <= 1)


class TestBioImageIOModelZoo:
    """Test the BioImage.IO model zoo"""

    def test_get_3D_model_by_id(self):
        """Try to load a 3D model from the BioImage.IO model zoo.

        Load Qin Yu's nuclear segmentation model 'efficient-chipmunk'.
        """
        model, _, model_path = model_zoo.get_model_by_id('efficient-chipmunk')
        state = torch.load(model_path, map_location='cpu', weights_only=True)
        if 'model_state_dict' in state:  # Model weights format may vary between versions
            state = state['model_state_dict']
        model.load_state_dict(state)

    def test_get_2D_model_by_id(self):
        """Try to load a 2D model from the BioImage.IO model zoo.

        Load Adrian's 2D cell-wall segmentation model 'pioneering-rhino'.
        """
        model, _, model_path = model_zoo.get_model_by_id('pioneering-rhino')
        state = torch.load(model_path, map_location='cpu', weights_only=True)
        if 'model_state_dict' in state:  # Model weights format may vary between versions
            state = state['model_state_dict']
        model.load_state_dict(state)

    def test_halo_computation_for_bioimageio_model(self):
        """Compute the halo for a BioImage.IO model."""
        model, _, _ = model_zoo.get_model_by_id('efficient-chipmunk')
        halo = model_zoo.compute_halo(model)
        assert halo == 44
