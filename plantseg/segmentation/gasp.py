import time

import numpy as np
from GASP.segmentation import GaspFromAffinities, WatershedOnDistanceTransformFromAffinities
from GASP.segmentation.watershed import SizeThreshAndGrowWithWS

from plantseg.pipeline import gui_logger
from plantseg.pipeline.steps import AbstractSegmentationStep
from plantseg.segmentation.utils import shift_affinities


class GaspFromPmaps(AbstractSegmentationStep):
    def __init__(self,
                 predictions_paths,
                 save_directory="GASP",
                 gasp_linkage_criteria='average',
                 bias=0.5,
                 run_ws=True,
                 ws_2D=True,
                 ws_threshold=0.4,
                 ws_minsize=50,
                 ws_sigma=0.3,
                 post_minsize=100,
                 n_threads=6,
                 **kwargs):

        super().__init__(input_paths=predictions_paths,
                         save_directory=save_directory,
                         file_suffix='_gasp_' + gasp_linkage_criteria,
                         num_threads=n_threads)

        assert gasp_linkage_criteria in ['average',
                                         'mutex_watershed'], f"Unsupported linkage criteria '{gasp_linkage_criteria}'"

        # GASP parameters
        self.gasp_linkage_criteria = gasp_linkage_criteria
        self.bias = bias

        # Watershed parameters
        self.run_ws = run_ws
        self.ws_2d = ws_2D
        self.ws_threshold = ws_threshold
        self.ws_minsize = ws_minsize
        self.ws_sigma = ws_sigma

        # Post processing size threshold
        self.post_minsize = post_minsize

        self.n_threads = n_threads

    def process(self, pmaps):
        gui_logger.info('Clustering with GASP...')

        # Pmaps are interpreted as affinities
        affinities = np.stack([pmaps, pmaps, pmaps], axis=0)

        offsets = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
        # Shift is required to correct aligned affinities
        affinities = shift_affinities(affinities, offsets=offsets)

        # invert affinities
        affinities = 1 - affinities

        # Run GASP:
        if self.run_ws:
            # In this case the agglomeration is initialized with superpixels:
            # use additional option 'intersect_with_boundary_pixels' to break the SP along the boundaries
            # (see CREMI-experiments script for an example)
            superpixel_gen = WatershedOnDistanceTransformFromAffinities(offsets,
                                                                        threshold=self.ws_threshold,
                                                                        min_segment_size=self.ws_minsize,
                                                                        preserve_membrane=True,
                                                                        sigma_seeds=self.ws_sigma,
                                                                        stacked_2d=not self.ws_2d,
                                                                        used_offsets=[0, 1, 2],
                                                                        offset_weights=[1, 1, 1],
                                                                        n_threads=self.n_threads)

        else:
            superpixel_gen = None

        # start real world clock timer
        runtime = time.time()

        run_GASP_kwargs = {'linkage_criteria': self.gasp_linkage_criteria,
                           'add_cannot_link_constraints': False,
                           'use_efficient_implementations': False}

        # Init and run Gasp
        gasp_instance = GaspFromAffinities(offsets,
                                           superpixel_generator=superpixel_gen,
                                           run_GASP_kwargs=run_GASP_kwargs,
                                           n_threads=self.n_threads,
                                           beta_bias=self.bias)
        # running gasp
        segmentation, _ = gasp_instance(affinities)

        # init and run size threshold
        size_threshold = SizeThreshAndGrowWithWS(self.post_minsize, offsets)
        segmentation = size_threshold(affinities, segmentation)

        # stop real world clock timer
        runtime = time.time() - runtime
        gui_logger.info(f"Clustering took {runtime:.2f} s")

        return segmentation
