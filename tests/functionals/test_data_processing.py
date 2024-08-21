import numpy as np

from plantseg.functionals.dataprocessing import (
    compute_scaling_factor,
    compute_scaling_voxelsize,
    fix_input_shape,
    image_crop,
    image_gaussian_smoothing,
    image_median,
    image_rescale,
    normalize_01,
    scale_image_to_voxelsize,
    select_channel,
)
from plantseg.functionals.dataprocessing.labelprocessing import relabel_segmentation, set_background_to_value


class TestDataProcessing:
    def test_compute_scaling_factor(self):
        input_voxel_size = (1.0, 1.0, 1.0)
        output_voxel_size = (0.5, 0.5, 0.5)
        scaling = compute_scaling_factor(input_voxel_size, output_voxel_size)
        assert scaling == (2.0, 2.0, 2.0)

    def test_compute_scaling_voxelsize(self):
        input_voxel_size = (1.0, 1.0, 1.0)
        scaling_factor = (2.0, 2.0, 2.0)
        output_voxel_size = compute_scaling_voxelsize(input_voxel_size, scaling_factor)
        assert output_voxel_size == (0.5, 0.5, 0.5)

    def test_image_to_voxelsize(self):
        image = np.random.rand(10, 10, 10)
        input_voxel_size = (1.0, 1.0, 1.0)
        output_voxel_size = (0.5, 0.5, 0.5)
        scaled_image = scale_image_to_voxelsize(image, input_voxel_size, output_voxel_size)
        assert scaled_image.shape == (20, 20, 20)

    def test_image_rescale(self):
        image = np.random.rand(10, 10, 10)
        factor = (2.0, 2.0, 2.0)
        scaled_image = image_rescale(image, factor, order=0)
        assert scaled_image.shape == (20, 20, 20)

    def test_smoothings(self):
        image = np.random.rand(10, 10, 10)
        median_image = image_median(image, 2)
        assert median_image.shape == (10, 10, 10)

        g_image = image_gaussian_smoothing(image, 1.0)
        assert g_image.shape == (10, 10, 10)

    def test_image_crop(self):
        image = np.random.rand(10, 10, 10)
        cropped_image = image_crop(image, "[:, 1:9, 1:-1]")
        assert cropped_image.shape == image[:, 1:9, 1:-1].shape

    def test_fix_input_shape(self):
        for shape_in, shape_out in [
            ((10, 10), (1, 10, 10)),
            ((10, 10, 10), (10, 10, 10)),
            ((1, 10, 10, 10), (10, 10, 10)),
        ]:
            image = np.random.rand(*shape_in)
            assert fix_input_shape(image, ndim=3).shape == shape_out

        for shape_in, shape_out in [((10, 10, 10), (10, 1, 10, 10)), ((2, 10, 10, 10), (2, 10, 10, 10))]:
            image = np.random.rand(*shape_in)
            assert fix_input_shape(image, ndim=4).shape == shape_out

    def test_normalize_01(self):
        image = np.random.rand(10, 10)
        normalized_image = normalize_01(image)
        assert normalized_image.min() >= 0.0
        assert normalized_image.max() <= 1.0 + 1e-6

    def test_relabel_segmentation(self):
        segmentation = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 3], [0, 0, 0, 3]])
        relabeled_segmentation = relabel_segmentation(segmentation)
        assert np.allclose(np.unique(relabeled_segmentation), [0, 1, 2])

    def test_set_background_to_value(self):
        segmentation = np.ones((10, 10))
        segmentation[1, 1] = 2
        segmentation[5, 5] = 3
        new_segmentation = set_background_to_value(segmentation, 0)
        assert np.allclose(np.unique(new_segmentation), [0, 3, 4])

    def test_select_channel(self):
        for shape_in, shape_out, ax in [
            ((2, 10, 10), (10, 10), 0),
            ((2, 10, 10, 10), (10, 10, 10), 0),
            ((10, 2, 10, 10), (10, 10, 10), 1),
        ]:
            image = np.random.rand(*shape_in)
            selected_image = select_channel(image, 0, channel_axis=ax)
            assert selected_image.shape == shape_out
