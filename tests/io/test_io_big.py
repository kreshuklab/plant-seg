from uuid import uuid4

import numpy as np
import pytest
from napari.layers import Image

from panseg.core.image import SemanticType
from panseg.io.tiff import create_tiff
from panseg.io.voxelsize import VoxelSize


@pytest.fixture
def napari_big_3d():
    data = np.empty((875, 760, 1700))
    voxel_size = None
    metadata = {
        "semantic_type": SemanticType.RAW,
        "voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "original_voxel_size": {"voxels_size": voxel_size, "unit": "um"},
        "image_layout": "YX",
        "id": uuid4(),
    }
    return Image(data, metadata=metadata, name="test_image_2D")


def test_big_image_small(tmp_path):
    data = np.empty((875, 760, 100), dtype="float32")
    out = tmp_path / "out.tiff"
    create_tiff(out, data, VoxelSize())
    assert out.exists()


def test_big_image_4G(tmp_path):
    data = np.empty((875, 760, 1800), dtype="float32")
    out = tmp_path / "out.tiff"
    create_tiff(out, data, VoxelSize())
    assert out.exists()
