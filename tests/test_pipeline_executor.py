import os
import queue

import h5py
import numpy as np
import pytest
import yaml

from plantseg.pipeline.executor import PipelineExecutor

TEST_FILES = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'resources',
)


@pytest.fixture
def input_path(tmpdir):
    path = os.path.join(tmpdir, 'test.h5')
    with h5py.File(path, 'w') as f:
        f.create_dataset('raw', data=np.random.randn(32, 128, 128))
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


class TestPipelineExecutor:
    def test_preprocessing(self, preprocess_config):
        executor = PipelineExecutor(max_workers=1, max_size=1)

        f = executor.submit(preprocess_config)
        # wait for the pipeline to finish
        f.result()
        # assert that work_queue is empty after the pipeline is done
        assert not executor.full()
        # clean up
        executor.shutdown()

    def test_full(self, preprocess_config):
        executor = PipelineExecutor(max_workers=1, max_size=1)

        f = executor.submit(preprocess_config)

        # assert that work_queue is full
        assert executor.full()

        # wait and clean up
        f.result()
        executor.shutdown()

    def test_full_work_queue_exception(self, preprocess_config):
        executor = PipelineExecutor(max_workers=1, max_size=1)
        with pytest.raises(queue.Full):
            f = executor.submit(preprocess_config)
            executor.submit(preprocess_config)

        # wait for the task to finish and cleanup
        f.result()
        executor.shutdown()
