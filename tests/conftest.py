# pylint: disable=missing-docstring,import-outside-toplevel

from pathlib import Path
import shutil

import h5py
import zarr
import numpy as np
import pytest
import yaml

TEST_FILES = Path(__file__).resolve().parent / 'resources'
VOXEL_SIZE = (0.235, 0.15, 0.15)
KEY_ZARR = 'volumes/new'


@pytest.fixture
def path_file_hdf5(tmpdir):
    """Create an HDF5 file using `h5py`'s API with an example dataset for testing purposes."""
    path = Path(tmpdir) / 'test.h5'
    with h5py.File(path, 'w') as f:
        f.create_dataset('raw', data=np.random.rand(32, 128, 128))
        f['raw'].attrs['element_size_um'] = VOXEL_SIZE
        f.create_dataset('segmentation', data=np.random.randint(low=0, high=256, size=(32, 128, 128)))
        f['segmentation'].attrs['element_size_um'] = VOXEL_SIZE
    return str(path)


@pytest.fixture
def path_file_zarr(tmpdir):
    """Create a Zarr file using `zarr`'s API with an example dataset for testing purposes."""

    path = Path(tmpdir) / 'test.zarr'
    zarr.save_array(str(path), np.random.rand(32, 128, 128), path=KEY_ZARR)
    zarr.open_array(path / KEY_ZARR, mode='a').attrs['element_size_um'] = VOXEL_SIZE
    return path


@pytest.fixture
def preprocess_config(path_file_hdf5):
    """Create pipeline config with only pre-processing (Gaussian filter) enabled."""
    config_path = TEST_FILES / 'test_config.yaml'
    config = yaml.full_load(config_path.read_text())
    # Add the file path to process
    config['path'] = path_file_hdf5
    # Enable Gaussian smoothing for some work
    config['preprocessing']['state'] = True
    config['preprocessing']['filter']['state'] = True
    return config


@pytest.fixture
def prediction_config(tmpdir):
    """Create pipeline config with Unet predictions enabled.

    Predictions will be executed on the `tests/resources/sample_ovules.h5`.
    The `sample_ovules.h5` file is copied to the temporary directory to avoid
    creating unnecessary files in `tests/resources`.
    """
    # Load the test configuration
    config_path = TEST_FILES / 'test_config.yaml'
    config = yaml.full_load(config_path.read_text())
    # Enable Unet predictions
    config['cnn_prediction']['state'] = True
    # Copy `sample_ovule.h5` to the temporary directory
    sample_ovule_path = TEST_FILES / 'sample_ovule.h5'
    tmp_path = Path(tmpdir) / 'sample_ovule.h5'
    shutil.copy2(sample_ovule_path, tmp_path)
    # Add the temporary path to the config
    config['path'] = str(tmp_path)  # Ensure the path is a string
    return config
