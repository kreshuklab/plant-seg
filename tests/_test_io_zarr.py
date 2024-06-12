"""
Tests the functionality of the `io.zarr` module.
"""

# pylint: disable=missing-docstring,import-outside-toplevel
import pytest

pytest.skip(allow_module_level=True)
from pathlib import Path
import numpy as np
import zarr

from tests.conftest import KEY_ZARR


class TestZarr:
    def test_create_zarr(self, tmpdir):
        from plantseg.io.zarr import create_zarr

        # There should be no files before the test
        tmpdir = Path(tmpdir)  # pytest fixture
        tmpfile_native = tmpdir / "test_native.zarr"
        tmpfile_plantseg = tmpdir / "test_plantseg.zarr"
        assert not tmpfile_native.exists(), f"File {tmpfile_native} already exists before test"
        assert not tmpfile_plantseg.exists(), f"File {tmpfile_plantseg} already exists before test"

        # 1. Native write to zarr file
        zarr.save_array(str(tmpfile_native), np.ones((32, 32, 32)), path=KEY_ZARR)  # only save needs str
        zarr.open_array(tmpfile_native / KEY_ZARR, mode="a").attrs["element_size_um"] = (1.0, 1.0, 1.0)

        # 2. PlantSeg write to zarr file with explicit voxel size
        create_zarr(str(tmpfile_plantseg), np.ones((32, 32, 32)), KEY_ZARR, voxel_size=(1.0, 1.0, 1.0), mode="a")

        # 3. PlantSeg write to zarr file without explicit voxel size
        create_zarr(str(tmpfile_plantseg), np.ones((32, 32, 32)), KEY_ZARR + "2")

        # Check if files were created and have the same content and voxel size
        zarr_array_native = zarr.open_array(tmpfile_native / KEY_ZARR, "r")
        zarr_array_plantseg = zarr.open_array(tmpfile_plantseg / KEY_ZARR, "r")
        zarr_array_plantseg2 = zarr.open_array(tmpfile_plantseg / (KEY_ZARR + "2"), "r")

        # fmt: off
        assert np.array_equal(zarr_array_native[:], np.ones((32, 32, 32))), "Data read from Zarr file is not equal to the original data"
        assert np.array_equal(zarr_array_native[:], zarr_array_plantseg[:]), "PlantSeg saved data is not equal to the native saved data"
        assert np.array_equal(zarr_array_native[:], zarr_array_plantseg2[:]), "PlantSeg saved data with default voxel size is not equal to the native saved data"
        assert tuple(zarr_array_native.attrs['element_size_um']) == (1.0, 1.0, 1.0), "Voxel size read from Zarr file is not equal to the original voxel size"
        assert zarr_array_native.attrs['element_size_um'] == zarr_array_plantseg.attrs['element_size_um'], "Voxel size from PlantSeg saved data is not equal to the native saved data"
        assert zarr_array_native.attrs['element_size_um'] == zarr_array_plantseg2.attrs['element_size_um'], "Voxel size from PlantSeg saved data with default voxel size is not equal to the native saved data"
        # fmt: on

    def test_load_zarr(self, path_file_zarr):
        from plantseg.io.zarr import load_zarr

        # File load with native function
        file_array_native = zarr.open_array(path_file_zarr / KEY_ZARR, "r")
        voxel_size_native = file_array_native.attrs["element_size_um"]

        # file load with specific dataset
        file_array_plantseg, (voxel_size_plantseg, _, _, _) = load_zarr(path=str(path_file_zarr), key=KEY_ZARR)

        # file load, with dataset key=None
        file_array_plantseg_2, (voxel_size_plantseg_2, _, _, _) = load_zarr(path=str(path_file_zarr), key=None)

        # only info load
        voxel_size_plantseg_3, _, _, _ = load_zarr(path=path_file_zarr, key=KEY_ZARR, info_only=True)

        # Check if loaded data are all the same
        # fmt: off
        assert np.array_equal(file_array_native[:], file_array_plantseg[:]), "Data read from Zarr file is not equal to the original data"
        assert np.array_equal(file_array_native[:], file_array_plantseg_2[:]), "Data read from Zarr file with key=None is not equal to the original data"
        assert voxel_size_native == voxel_size_plantseg, "Voxel size read from Zarr file with a key is not equal to the original voxel size"
        assert voxel_size_native == voxel_size_plantseg_2, "Voxel size read from Zarr file with key=None is not equal to the original voxel size"
        assert voxel_size_native == voxel_size_plantseg_3, "Voxel size read from Zarr file with info_only=True is not equal to the original voxel size"
        # fmt: on

    def test_list_keys(self, tmpdir):
        from plantseg.io.zarr import list_keys

        tmpdir = Path(tmpdir)  # pytest fixture
        tmpfile = tmpdir / "test.zarr"

        # Note that this causes error: `keys = ['/group1/array0', '/group2/array1', '/group2/array2']``
        keys = ["array0", "group1/array1", "group2/array2", "group2/group3/array3"]
        for key in keys:
            zarr.save_array(str(tmpfile), np.ones((8, 8, 8)), path=key)
            zarr.open_array(tmpfile / key, mode="a").attrs["element_size_um"] = (1.0, 1.0, 1.0)
        assert list_keys(str(tmpfile)) == keys

    def test_rename_zarr_key(self, path_file_zarr):
        from plantseg.io.zarr import list_keys, rename_zarr_key

        key_new = "volumes_new/new"

        assert list_keys(path_file_zarr) == [KEY_ZARR]
        rename_zarr_key(path_file_zarr, KEY_ZARR, key_new)
        assert list_keys(path_file_zarr) == [key_new]  # Note that the old but now empty group still exist

    def test_del_zarr_key(self, path_file_zarr):
        from plantseg.io.zarr import list_keys, del_zarr_key

        assert list_keys(path_file_zarr) == [KEY_ZARR]
        del_zarr_key(path_file_zarr, KEY_ZARR)
        assert list_keys(path_file_zarr) == []
