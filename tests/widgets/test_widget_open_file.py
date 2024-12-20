import os

import napari
import numpy as np
import pytest

from plantseg.io.h5 import create_h5
from plantseg.io.voxelsize import VoxelSize
from plantseg.viewer_napari.widgets.io import PathMode, widget_open_file

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"  # set to true in GitHub Actions by default to skip CUDA tests


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="GUI tests hangs in GitHub Actions.")
def test_widget_open_file(make_napari_viewer_proxy, path_h5):
    viewer = make_napari_viewer_proxy()
    shape = (10, 10, 10)
    data = np.random.rand(*shape).astype(np.float32)
    voxel_size = VoxelSize(voxels_size=(0.235, 0.15, 0.15))
    create_h5(path_h5, data, "raw", voxel_size=voxel_size)
    create_h5(path_h5, data, "prob", voxel_size=voxel_size)

    widget_open_file(
        path_mode=PathMode.FILE.value,
        path=path_h5,
        layer_type="image",
        new_layer_name="test_raw",
        dataset_key="/raw",
        stack_layout="ZYX",
        update_other_widgets=False,
    )
    napari.run()

    assert viewer.layers[0].name == "test_raw"
    assert viewer.layers[0].data.shape == shape
    assert viewer.layers[0].data.dtype == data.dtype
    np.testing.assert_array_equal(viewer.layers[0].scale, (voxel_size.z, voxel_size.y, voxel_size.x))
