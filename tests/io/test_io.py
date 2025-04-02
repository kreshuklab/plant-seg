import numpy as np
import pytest

from plantseg.io.io import smart_load
from plantseg.io.voxelsize import VoxelSize


class TestIO:
    voxel_size = VoxelSize(voxels_size=(0.235, 0.15, 0.15))

    def _check_voxel_size(self, voxel_size):
        assert isinstance(voxel_size, VoxelSize)
        np.testing.assert_allclose(
            voxel_size,
            self.voxel_size,
            rtol=1e-5,
            err_msg="Voxel size read from file is not equal to the original voxel size",
        )

    def test_create_read_h5(self, path_h5):
        from plantseg.io.h5 import create_h5, load_h5, read_h5_shape, read_h5_voxel_size

        # Create an HDF5 file
        data = np.random.rand(10, 10, 10)
        create_h5(path_h5, data, "raw", voxel_size=self.voxel_size)

        # Create an HDF5 file with empty name causes an error
        with pytest.raises(ValueError):
            create_h5(path_h5, data, "", voxel_size=self.voxel_size)
        with pytest.raises(ValueError):
            create_h5(path_h5, data, None, voxel_size=self.voxel_size)

        # Read the HDF5 file
        data_read = load_h5(path_h5, "raw")
        assert np.array_equal(data, data_read), (
            "Data read from HDF5 file is not equal to the original data"
        )

        # Read the shape of the HDF5 file
        shape = read_h5_shape(path_h5, "raw")
        assert shape == data.shape, (
            "Shape read from HDF5 file is not equal to the original shape"
        )

        # Read the voxel size of the HDF5 file
        voxel_size = read_h5_voxel_size(path_h5, "raw")
        self._check_voxel_size(voxel_size)

        data_read2 = smart_load(path_h5, "raw")
        assert np.array_equal(data, data_read2), (
            "Data read from HDF5 file is not equal to the original data"
        )

    def test_create_read_zarr(self, path_zarr):
        from plantseg.io.zarr import (
            create_zarr,
            load_zarr,
            read_zarr_shape,
            read_zarr_voxel_size,
        )

        # Create a Zarr file
        data = np.random.rand(10, 10, 10)
        create_zarr(path_zarr, data, "raw", voxel_size=self.voxel_size)

        # Create a Zarr file with empty name causes an error
        with pytest.raises(ValueError):
            create_zarr(path_zarr, data, "", voxel_size=self.voxel_size)
        with pytest.raises(ValueError):
            create_zarr(path_zarr, data, None, voxel_size=self.voxel_size)

        # Read the Zarr file
        data_read = load_zarr(path_zarr, "raw")
        assert np.array_equal(data, data_read), (
            "Data read from Zarr file is not equal to the original data"
        )

        # Read the shape of the Zarr file
        shape = read_zarr_shape(path_zarr, "raw")
        assert shape == data.shape, (
            "Shape read from Zarr file is not equal to the original shape"
        )

        # Read the voxel size of the Zarr file
        voxel_size = read_zarr_voxel_size(path_zarr, "raw")
        assert np.allclose(voxel_size, self.voxel_size), (
            "Voxel size read from Zarr file is not equal to the original voxel size"
        )

        data_read2 = smart_load(path_zarr, "raw")
        assert np.array_equal(data, data_read2), (
            "Data read from Zarr file is not equal to the original data"
        )

    def test_create_read_tiff(self, path_tiff):
        from plantseg.io.tiff import create_tiff, load_tiff, read_tiff_voxel_size

        # Create a TIFF file
        data = 255 * np.random.rand(10, 10, 10)
        data = data.astype(np.uint8)
        create_tiff(path_tiff, data, voxel_size=self.voxel_size)

        # Read the TIFF file
        data_read = load_tiff(path_tiff)
        assert np.array_equal(data, data_read), (
            "Data read from TIFF file is not equal to the original data"
        )

        # Read the voxel size of the TIFF file
        voxel_size = read_tiff_voxel_size(path_tiff)
        self._check_voxel_size(voxel_size)

        data_read2 = smart_load(path_tiff)
        assert np.array_equal(data, data_read2), (
            "Data read from TIFF file is not equal to the original data"
        )

    def test_read_jpg(self, path_jpg):
        import numpy as np
        from PIL import Image

        from plantseg.io.pil import load_pil

        # Create a JPG file
        data = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        image = Image.fromarray(data)
        image.save(path_jpg)

        # Implicitly convrrted from RGB to grayscale
        data_read = load_pil(path_jpg)

        assert data.shape[:2] == data_read.shape, (
            "Data read from JPG file is not equal to the original data"
        )
