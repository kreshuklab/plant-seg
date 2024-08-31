import numpy as np
import pytest

from plantseg.functionals.dataprocessing.dataprocessing import (
    compute_scaling_factor,
    compute_scaling_voxelsize,
    fix_input_shape,
    fix_input_shape_to_CZYX,
    fix_input_shape_to_ZYX,
    image_crop,
    image_gaussian_smoothing,
    image_median,
    image_rescale,
    normalize_01,
    normalize_01_channel_wise,
    scale_image_to_voxelsize,
    select_channel,
)


# Test compute_scaling_factor
def test_compute_scaling_factor():
    input_voxel_size = (2.0, 2.0, 2.0)
    output_voxel_size = (1.0, 1.0, 1.0)
    scaling = compute_scaling_factor(input_voxel_size, output_voxel_size)
    assert scaling == (2.0, 2.0, 2.0)

    with pytest.raises(ValueError):
        compute_scaling_factor((2.0, 2.0), (1.0, 1.0, 1.0))


# Test compute_scaling_voxelsize
def test_compute_scaling_voxelsize():
    input_voxel_size = (2.0, 2.0, 2.0)
    scaling_factor = (2.0, 2.0, 2.0)
    output_voxel_size = compute_scaling_voxelsize(input_voxel_size, scaling_factor)
    assert output_voxel_size == (1.0, 1.0, 1.0)

    with pytest.raises(ValueError):
        compute_scaling_voxelsize((2.0, 2.0), (2.0, 2.0, 2.0))


# Test scale_image_to_voxelsize
def test_scale_image_to_voxelsize():
    image = np.random.rand(10, 10, 10)
    input_voxel_size = (2.0, 2.0, 2.0)
    output_voxel_size = (1.0, 1.0, 1.0)
    scaled_image = scale_image_to_voxelsize(image, input_voxel_size, output_voxel_size)
    assert scaled_image.shape == (20, 20, 20)


# Test image_rescale
def test_image_rescale():
    image = np.random.rand(10, 10, 10)
    factor = (2.0, 2.0, 2.0)
    rescaled_image = image_rescale(image, factor, order=1)
    assert rescaled_image.shape == (20, 20, 20)

    # No rescaling (factor is 1)
    rescaled_image = image_rescale(image, (1.0, 1.0, 1.0), order=1)
    assert np.array_equal(rescaled_image, image)


# Test image_median
def test_image_median():
    radius = 1

    image = np.random.rand(10, 10)
    median_image = image_median(image, radius)
    assert median_image.shape == (10, 10)

    image = np.random.rand(1, 10, 10)
    median_image = image_median(image, radius)
    assert median_image.shape == (1, 10, 10)

    image_3d = np.random.rand(10, 10, 10)
    median_image_3d = image_median(image_3d, radius)
    assert median_image_3d.shape == (10, 10, 10)


# Test image_gaussian_smoothing
def test_image_gaussian_smoothing():
    image = np.random.rand(10, 10, 10)
    sigma = 1.0
    smoothed_image = image_gaussian_smoothing(image, sigma)
    assert smoothed_image.shape == (10, 10, 10)
    assert smoothed_image.dtype == np.float32


# Test image_crop
def test_image_crop():
    image = np.random.rand(10, 10, 10)
    cropped_image = image_crop(image, "[2:8, 2:8, 2:8]")
    assert cropped_image.shape == (6, 6, 6)

    cropped_image = image_crop(image, "[2:, :8, :8]")
    assert cropped_image.shape == (8, 8, 8)


# Test fix_input_shape
def test_fix_input_shape():
    for shape_in, shape_out in [
        ((10, 10), (1, 10, 10)),
        ((10, 10, 10), (10, 10, 10)),
        ((1, 10, 10, 10), (10, 10, 10)),
    ]:
        image = np.random.rand(*shape_in)
        assert fix_input_shape(image, ndim=3).shape == shape_out

    for shape_in, shape_out in [
        ((10, 10, 10), (10, 1, 10, 10)),
        ((2, 10, 10, 10), (2, 10, 10, 10)),
    ]:
        image = np.random.rand(*shape_in)
        assert fix_input_shape(image, ndim=4).shape == shape_out


# Test fix_input_shape_to_ZYX
def test_fix_input_shape_to_ZYX():
    data_2d = np.random.rand(10, 10)
    fixed_data = fix_input_shape_to_ZYX(data_2d)
    assert fixed_data.shape == (1, 10, 10)

    data_3d = np.random.rand(10, 10, 10)
    fixed_data = fix_input_shape_to_ZYX(data_3d)
    assert fixed_data.shape == (10, 10, 10)

    data_4d = np.random.rand(5, 10, 10, 10)
    fixed_data = fix_input_shape_to_ZYX(data_4d)
    assert fixed_data.shape == (10, 10, 10)


# Test fix_input_shape_to_CZYX
def test_fix_input_shape_to_CZYX():
    data_3d = np.random.rand(10, 10, 10)
    fixed_data = fix_input_shape_to_CZYX(data_3d)
    assert fixed_data.shape == (10, 1, 10, 10)

    data_4d = np.random.rand(2, 5, 10, 10)
    fixed_data = fix_input_shape_to_CZYX(data_4d)
    assert fixed_data.shape == (2, 5, 10, 10)


# Test normalize_01
def test_normalize_01():
    data = np.random.rand(10, 10, 10)
    normalized_data = normalize_01(data)
    assert np.allclose(np.min(normalized_data), 0)
    assert np.allclose(np.max(normalized_data), 1)
    assert normalized_data.min() >= 0.0
    assert normalized_data.max() <= 1.0 + 1e-6


# Test select_channel
def test_select_channel():
    data = np.random.rand(5, 10, 10, 10)

    channel_data = select_channel(data, 2, channel_axis=0)
    assert channel_data.shape == (10, 10, 10)

    channel_data = select_channel(data, 2, channel_axis=1)
    assert channel_data.shape == (5, 10, 10)


# Test normalize_01_channel_wise
def test_normalize_01_channel_wise():
    data = np.random.rand(3, 10, 10, 10)
    normalized_data = normalize_01_channel_wise(data, channel_axis=0)
    assert normalized_data.shape == (3, 10, 10, 10)
    for i in range(3):
        assert np.allclose(np.min(normalized_data[i]), 0)
        assert np.allclose(np.max(normalized_data[i]), 1)
