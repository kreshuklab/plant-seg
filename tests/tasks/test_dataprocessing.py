import numpy as np
import pytest

from plantseg.core.image import PlantSegImage
from plantseg.io.voxelsize import VoxelSize
from plantseg.tasks import dataprocessing_tasks as dpt


def test_compute_slices_3d_no_rectangle(raw_cell_3d_100x128x128):
    """Test slicing with no rectangle provided."""
    shape = raw_cell_3d_100x128x128.shape
    z_slice, x_slice, y_slice = dpt._compute_slices_3d(None, (10, 50), shape)

    assert z_slice == slice(10, 50)
    assert x_slice == slice(0, shape[1])
    assert y_slice == slice(0, shape[2])


def test_compute_slices_3d_with_broken_rectangle(raw_cell_3d_100x128x128):
    """Test slicing with a valid rectangle."""
    shape = raw_cell_3d_100x128x128.shape
    rectangle = np.array([[10, 20, 30], [10, 20, 30], [50, 80, 90], [50, 80, 90]])
    with pytest.raises(ValueError):
        z_slice, x_slice, y_slice = dpt._compute_slices_3d(rectangle, (0, 30), shape)


def test_compute_slices_3d_with_rectangle(raw_cell_3d_100x128x128):
    """Test slicing with a valid rectangle."""
    shape = raw_cell_3d_100x128x128.shape
    rectangle = np.array([[10, 10, 10], [10, 10, 30], [10, 80, 30], [10, 80, 10]])

    z_slice, x_slice, y_slice = dpt._compute_slices_3d(rectangle, (0, 30), shape)

    assert z_slice == slice(0, 30)
    assert x_slice == slice(10, 80)
    assert y_slice == slice(10, 30)


def test_compute_slices_2d_no_rectangle():
    """Test slicing with None rectangle returns full slices."""
    shape = (100, 100)
    x_slice, y_slice = dpt._compute_slices_2d(None, shape)
    assert x_slice == slice(0, 100)
    assert y_slice == slice(0, 100)


def test_compute_slices_2d_with_rectangle():
    """Test slicing with valid rectangle coordinates."""
    rectangle = np.array(
        [[10, 20], [0, 0], [50, 60]]
    )  # [x_start, y_start], [0, 0], [x_end, y_end]
    shape = (100, 100)
    x_slice, y_slice = dpt._compute_slices_2d(rectangle, shape)
    assert x_slice == slice(10, 50)
    assert y_slice == slice(20, 60)


def test_compute_slices_2d_clamping():
    """Test that coordinates are clamped to image boundaries."""
    rectangle = np.array([[-10, -5], [0, 0], [150, 120]])  # Out of bounds coordinates
    shape = (100, 100)
    x_slice, y_slice = dpt._compute_slices_2d(rectangle, shape)
    assert x_slice == slice(0, 100)
    assert y_slice == slice(0, 100)


def test_cropping():
    np.testing.assert_equal(dpt._cropping(np.zeros((3, 3)), np.array([0, 0])), 0)


def test_image_croppig(napari_raw):
    ps_image = PlantSegImage.from_napari_layer(napari_raw)
    dpt.image_cropping_task(image=ps_image)


def test_set_voxel_size_task(napari_raw):
    ps_image = PlantSegImage.from_napari_layer(napari_raw)
    out = dpt.set_voxel_size_task(image=ps_image, voxel_size=(0.1, 0.2, 0.3))
    assert out.voxel_size == VoxelSize(voxels_size=(0.1, 0.2, 0.3))


def test_image_rescale_to_shape_task_yx(napari_raw_2d):
    ps_image = PlantSegImage.from_napari_layer(napari_raw_2d)
    out = dpt.image_rescale_to_shape_task(image=ps_image, new_shape=(1, 100, 200))
    assert out.shape == (100, 200)


def test_image_rescale_to_shape_task_zyx(napari_raw):
    ps_image = PlantSegImage.from_napari_layer(napari_raw)
    out = dpt.image_rescale_to_shape_task(image=ps_image, new_shape=(10, 20, 30))
    assert out.shape == (10, 20)
