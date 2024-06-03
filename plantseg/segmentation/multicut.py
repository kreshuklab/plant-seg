import time
from functools import partial

from elf.segmentation.watershed import apply_size_filter

from plantseg.pipeline import gui_logger
from plantseg.pipeline.steps import AbstractSegmentationStep
from plantseg.segmentation.functional.segmentation import dt_watershed, multicut


class MulticutFromPmaps(AbstractSegmentationStep):
    def __init__(
        self,
        predictions_paths,
        key=None,
        channel=None,
        save_directory="MultiCut",
        beta=0.5,
        run_ws=True,
        ws_2D=True,
        ws_threshold=0.4,
        ws_minsize=50,
        ws_sigma=2.0,
        ws_w_sigma=0,
        post_minsize=50,
        n_threads=6,
        state=True,
        **kwargs,
    ):
        super().__init__(
            input_paths=predictions_paths,
            save_directory=save_directory,
            file_suffix='_multicut',
            state=state,
            input_key=key,
            input_channel=channel,
        )

        self.beta = beta

        # Watershed parameters
        self.run_ws = run_ws
        self.ws_2D = ws_2D
        self.ws_threshold = ws_threshold
        self.ws_minsize = ws_minsize
        self.ws_sigma = ws_sigma
        self.ws_w_sigma = ws_w_sigma

        # Postprocessing size threshold
        self.post_minsize = post_minsize

        # Multithread
        self.n_threads = n_threads

        self.dt_watershed = partial(
            dt_watershed,
            threshold=ws_threshold,
            sigma_seeds=ws_sigma,
            stacked=ws_2D,
            sigma_weights=ws_w_sigma,
            min_size=ws_minsize,
            n_threads=n_threads,
        )

    def process(self, pmaps):
        runtime = time.time()
        gui_logger.info('Computing segmentation with dtWS...')
        ws = self.dt_watershed(pmaps)

        gui_logger.info('Clustering with MultiCut...')
        segmentation = multicut(pmaps, superpixels=ws, beta=self.beta, post_minsize=self.post_minsize)

        if self.post_minsize > self.ws_minsize:
            segmentation, _ = apply_size_filter(segmentation, pmaps, self.post_minsize)

        # stop real world clock timer
        runtime = time.time() - runtime
        gui_logger.info(f"Clustering took {runtime:.2f} s")

        return segmentation
