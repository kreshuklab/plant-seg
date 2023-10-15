import time
from functools import partial

from plantseg.pipeline import gui_logger
from plantseg.pipeline.steps import AbstractSegmentationStep
from plantseg.segmentation.functional.segmentation import dt_watershed, gasp


class WSSegmentationFeeder:
    def __init__(self, segmentation):
        self.segmentation = segmentation

    def __call__(self, *args, **kwargs):
        return self.segmentation


class GaspFromPmaps(AbstractSegmentationStep):
    def __init__(self,
                 predictions_paths,
                 key=None,
                 channel=None,
                 save_directory="GASP",
                 gasp_linkage_criteria='average',
                 beta=0.5,
                 run_ws=True,
                 ws_2D=True,
                 ws_threshold=0.4,
                 ws_minsize=50,
                 ws_sigma=0.3,
                 ws_w_sigma=0,
                 post_minsize=100,
                 n_threads=6,
                 state=True,
                 **kwargs):

        super().__init__(input_paths=predictions_paths,
                         save_directory=save_directory,
                         file_suffix='_gasp_' + gasp_linkage_criteria,
                         state=state,
                         input_key=key,
                         input_channel=channel)

        assert gasp_linkage_criteria in ['average',
                                         'mutex_watershed'], f"Unsupported linkage criteria '{gasp_linkage_criteria}'"

        # GASP parameters
        self.gasp_linkage_criteria = gasp_linkage_criteria
        self.beta = beta

        # Watershed parameters
        self.run_ws = run_ws
        self.ws_2d = ws_2D
        self.ws_threshold = ws_threshold
        self.ws_minsize = ws_minsize
        self.ws_sigma = ws_sigma

        # Postprocessing size threshold
        self.post_minsize = post_minsize

        self.n_threads = n_threads

        self.dt_watershed = partial(dt_watershed,
                                    threshold=ws_threshold, sigma_seeds=ws_sigma,
                                    stacked=ws_2D, sigma_weights=ws_w_sigma,
                                    min_size=ws_minsize, n_threads=n_threads)

    def process(self, pmaps):
        # start real world clock timer
        runtime = time.time()

        if self.run_ws:
            # In this case the agglomeration is initialized with superpixels:
            # use additional option 'intersect_with_boundary_pixels' to break the SP along the boundaries
            # (see CREMI-experiments script for an example)
            gui_logger.info('Computing segmentation with dtWS...')
            ws = self.dt_watershed(pmaps)

        else:
            ws = None

        gui_logger.info('Clustering with GASP...')
        # Run GASP
        segmentation = gasp(pmaps, ws,
                            gasp_linkage_criteria=self.gasp_linkage_criteria,
                            beta=self.beta,
                            post_minsize=self.post_minsize,
                            n_threads=self.n_threads)

        # stop real world clock timer
        runtime = time.time() - runtime
        gui_logger.info(f"Clustering took {runtime:.2f} s")

        return segmentation
