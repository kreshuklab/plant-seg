import time
from functools import partial

from plantseg.pipeline import gui_logger
from plantseg.pipeline.steps import AbstractSegmentationStep
from plantseg.segmentation.functional.segmentation import dt_watershed


class DistanceTransformWatershed(AbstractSegmentationStep):
    def __init__(self,
                 predictions_paths,
                 save_directory="DTWatershed",
                 ws_2D=True,
                 ws_threshold=0.4,
                 ws_minsize=50,
                 ws_sigma=0.3,
                 ws_w_sigma=0,
                 n_threads=None,
                 state=True,
                 **kwargs):
        super().__init__(input_paths=predictions_paths,
                         save_directory=save_directory,
                         file_suffix='_dtws',
                         state=state)

        self.dt_watershed = partial(dt_watershed,
                                    threshold=ws_threshold, sigma_seeds=ws_sigma,
                                    stacked=ws_2D, sigma_weights=ws_w_sigma,
                                    min_size=ws_minsize, n_threads=n_threads)

    def process(self, pmaps):
        runtime = time.time()

        gui_logger.info('Computing segmentation with dtWS...')
        segmentation = self.dt_watershed(pmaps)

        # stop real world clock timer
        runtime = time.time() - runtime
        gui_logger.info(f"Segmentation took {runtime:.2f} s")
        return segmentation
