import os

import h5py
import numpy as np

from plantseg.dataprocessing.dataprocessing import DataPostProcessing3D, DataPreProcessing3D, _gaussian


class TestDataProcessing:
    def test_preprocessing(self, input_path):
        pre = DataPreProcessing3D([input_path], input_type="data_uint8", output_type="data_uint8",
                                  save_directory="PreProcessing", filter_type="gaussian", filter_param=1.0)

        # run preprocessing
        output_paths = pre()

        # assert output path correctness
        basepath, basename = os.path.split(input_path)
        assert output_paths[0] == os.path.join(basepath, "PreProcessing", basename)

        # assert content equal
        # TODO: this is currently quite convoluted due to _adjust_input_type/_adjust_output_type functions
        # with h5py.File(input_path, 'r') as f:
        #     raw = f['raw'][...]
        #
        # with h5py.File(output_paths[0], 'r') as f:
        #     raw_filtered = f['raw'][...]

    def test_preprocessing_crop(self, input_path):
        with h5py.File(input_path, 'r') as f:
            raw = f['raw'][...]

        target_shape = (raw.shape[0], 32, 32)
        pre = DataPreProcessing3D([input_path], input_type="data_uint8", output_type="data_uint8",
                                  save_directory="PreProcessing", filter_type=None, filter_param=None,
                                  crop='[:, :32, :32]')

        output_paths = pre()

        with h5py.File(output_paths[0], 'r') as f:
            raw = f['raw'][...]

        assert raw.shape == target_shape

    def test_postprocessing_tiff(self, input_path):
        post = DataPostProcessing3D([input_path], save_directory="PostProcessing", out_ext=".tiff")

        # run postprocessing
        output_paths = post()

        # assert output path correctness
        basepath, basename = os.path.split(input_path)
        assert output_paths[0] == os.path.join(basepath, "PostProcessing", os.path.splitext(basename)[0] + '.tiff')

    def test_postprocessing_factor(self, input_path):
        factor = [1, 2, 2]
        post = DataPostProcessing3D([input_path], save_directory="PostProcessing", factor=factor)

        # run postprocessing
        output_paths = post()

        # assert output path correctness
        basepath, basename = os.path.split(input_path)
        assert output_paths[0] == os.path.join(basepath, "PostProcessing", basename)

        # assert output shape
        with h5py.File(input_path, 'r') as f:
            original = f['segmentation'][...]

        with h5py.File(output_paths[0], 'r') as f:
            rescaled = f['segmentation'][...]

        assert np.allclose(np.array(rescaled.shape), np.array(original.shape) * factor)

    def test_processing_disabled(self, input_path):
        pre = DataPreProcessing3D([input_path], state=False)
        post = DataPostProcessing3D([input_path], state=False)

        # assert that output paths are equal to input paths
        assert pre() == [input_path]
        assert post() == [input_path]

    def test_preprocessing_default_voxel_size(self, tmpdir):
        path = os.path.join(tmpdir, 'test.h5')
        # create a new file without element_size_um
        with h5py.File(path, 'w') as f:
            f.create_dataset('raw', data=np.random.rand(32, 128, 128))

        pre = DataPreProcessing3D([path], input_type="data_uint8", output_type="data_uint8")
        # run preprocessing
        output_paths = pre()

        # check output voxel_size
        with h5py.File(output_paths[0], 'r') as f:
            voxel_size = f['raw'].attrs['element_size_um']

        assert np.allclose((1, 1, 1), voxel_size)

    def test_preprocessing_voxel_size(self, input_path):
        with h5py.File(input_path, 'r') as f:
            expected_voxel_size = f['raw'].attrs['element_size_um']

        pre = DataPreProcessing3D([input_path], input_type="data_uint8", output_type="data_uint8")
        # run preprocessing
        output_paths = pre()

        # check output voxel_size
        with h5py.File(output_paths[0], 'r') as f:
            voxel_size = f['raw'].attrs['element_size_um']

        assert np.allclose(np.array(expected_voxel_size), np.array(voxel_size))

    def test_tiff_voxel_size(self, input_path):
        """
        Take input h5, convert to tiff, convert to h5, check if the voxel size matches the original
        """
        # convert to h5 to tiff
        post = DataPostProcessing3D([input_path], input_type="labels", output_type="labels", out_ext='.tiff')
        output_paths = post()
        # convert tiff to h5
        pre = DataPreProcessing3D(output_paths, input_type="labels", output_type="labels")
        output_paths = pre()

        # check output voxel_size
        with h5py.File(input_path, 'r') as f:
            expected_voxel_size = f['raw'].attrs['element_size_um']

        with h5py.File(output_paths[0], 'r') as f:
            voxel_size = f['raw'].attrs['element_size_um']

        assert np.allclose(expected_voxel_size, voxel_size)
