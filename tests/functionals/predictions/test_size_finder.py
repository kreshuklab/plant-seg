import os

import pytest
import torch

from plantseg.core.zoo import model_zoo
from plantseg.functionals.predictions.utils.size_finder import find_patch_and_halo_shapes, find_patch_shape

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
MAX_PATCH_SHAPES = {
    'generic_confocal_3D_unet': {
        "NVIDIA GeForce RTX 2080 Ti": (208, 208, 208),
        "NVIDIA GeForce RTX 3090": (256, 256, 256),
    }
}
GPU_DEVICE_NAME = torch.cuda.get_device_name(0) if not IN_GITHUB_ACTIONS else ""


@pytest.mark.parametrize(
    "full_volume_shape, max_patch_shape, min_halo_shape, expected",
    [  # Here halo is 1-sided
        ((1, 12000, 50000), (192, 192, 192), (44, 44, 44), ((1, 2572, 2572), (0, 44, 44))),
        ((1, 120, 500), (192, 192, 192), (44, 44, 44), ((1, 120, 500), (0, 0, 0))),
        ((95, 120, 500), (192, 192, 192), (44, 44, 44), ((95, 120, 500), (0, 0, 0))),
        ((95, 120, 5000), (192, 192, 192), (44, 44, 44), ((95, 120, 532), (0, 0, 44))),
        ((5000, 120, 95), (192, 192, 192), (44, 44, 44), ((532, 120, 95), (44, 0, 0))),
        ((100, 1000, 1000), (192, 192, 192), (44, 44, 44), ((100, 178, 178), (0, 44, 44))),
        ((1000, 1000, 1000), (192, 192, 192), (44, 44, 44), ((104, 104, 104), (44, 44, 44))),
    ],
)
def test_find_patch_and_halo_shapes(full_volume_shape, max_patch_shape, min_halo_shape, expected):
    result = find_patch_and_halo_shapes(full_volume_shape, max_patch_shape, min_halo_shape)
    assert result == expected
    double_halo_shape = (min_halo_shape[0] * 2, min_halo_shape[1] * 2, min_halo_shape[2] * 2)
    result = find_patch_and_halo_shapes(full_volume_shape, max_patch_shape, double_halo_shape, both_sides=True)
    assert result == expected


@pytest.mark.skipif(
    GPU_DEVICE_NAME not in ["NVIDIA GeForce RTX 2080 Ti", "NVIDIA GeForce RTX 3090"],
    reason="Measured devices are not available.",
)
@pytest.mark.parametrize("model_name", MAX_PATCH_SHAPES.keys())
def test_find_patch_shape(model_name):
    model, _, _ = model_zoo.get_model_by_name(model_name, model_update=False)
    found_patch_shape = find_patch_shape(model, 1, "cuda:0")
    expected_patch_shape = MAX_PATCH_SHAPES[model_name][GPU_DEVICE_NAME]
    assert found_patch_shape == expected_patch_shape
