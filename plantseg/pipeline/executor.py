import queue
from concurrent import futures

from plantseg.main import gui_logger
from plantseg.main.raw2seg import raw2seg
from plantseg.pipeline.utils import get_logger

logger = get_logger('PipelineExecutor')


class PipelineExecutor:
    """
    The main class responsible for executing the pipeline. Takes care of running the pipeline tasks
    in separate threads not to block the UI, as well as limiting the number of concurrent pipeline tasks

    Args:
        max_workers (int): the maximum number of threads that can be used to
                execute the given calls; default: 1
        max_size (int): the maximum number of tasks a user can submit to the PipelineExecutor; default: 1
    """

    def __init__(self, max_workers=1, max_size=1):
        self.executor = futures.ThreadPoolExecutor(max_workers=max_workers)
        self.work_queue = queue.Queue(maxsize=max_size)

    def submit(self, config):
        """
        Executes the pipeline task described by the config. If the work_queue is full throws an exception.
        It is up to the user to check if the work_queue is full.
        """

        logger.info(f"Executing segmentation pipeline for config: {config}")
        # add config to the queue
        self.work_queue.put_nowait(config)
        # execute segmentation pipeline
        future = self.executor.submit(raw2seg, config)
        # remove the config from the queue when finished
        future.add_done_callback(self._done_callback)
        return future

    def _done_callback(self, f):
        self.work_queue.get_nowait()
        try:
            f.result()
        except Exception as e:
            # log exceptions from the pipeline in the console and in the GUI
            logger.exception(e)
            gui_logger.exception(e)

    def full(self):
        """
        Returns True if the work_queue is full
        """
        return self.work_queue.full()

    def shutdown(self, wait=False):
        logger.info("Shutting down")
        self.executor.shutdown(wait=wait)
