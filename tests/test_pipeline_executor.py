import queue

import pytest

from plantseg.pipeline.executor import PipelineExecutor


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
