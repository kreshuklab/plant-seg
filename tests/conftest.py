import os

import h5py
import numpy as np
import pytest
import yaml

TEST_FILES = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'resources',
)


@pytest.fixture
def input_path(tmpdir):
    path = os.path.join(tmpdir, 'test.h5')
    with h5py.File(path, 'w') as f:
        f.create_dataset('raw', data=np.random.rand(32, 128, 128))
        f.create_dataset('segmentation', data=np.random.randint(low=0, high=256, size=(32, 128, 128)))
    return path


@pytest.fixture
def preprocess_config(input_path):
    """
    Create pipeline config with only pre-processing (gaussian fileter) enabled
    """
    config_path = os.path.join(TEST_FILES, 'test_executor_config.yaml')
    config = yaml.full_load(open(config_path, 'r'))
    # add file to process
    config['path'] = input_path
    # add gaussian smoothing just to do some work
    config['preprocessing']['state'] = True
    config['preprocessing']['filter']['state'] = True
    return config
