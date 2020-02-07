import numpy as np
from GASP.segmentation import WatershedOnDistanceTransformFromAffinities

from plantseg.pipeline.steps import AbstractSegmentationStep
from plantseg.segmentation.utils import shift_affinities


class DistanceTransformWatershed(AbstractSegmentationStep):
    def __init__(self,
                 predictions_paths,
                 save_directory="DTWatershed",
                 ws_2D=True,
                 ws_threshold=0.4,
                 ws_minsize=50,
                 ws_sigma=0.3,
                 n_threads=8,
                 state=True,
                 **kwargs):
        super().__init__(input_paths=predictions_paths,
                         save_directory=save_directory,
                         file_suffix='_dtws',
                         state=state)

        self.offsets = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
        # TODO: should we use this DTWS or the one from elf
        # In this case the agglomeration is initialized with superpixels:
        # use additional option 'intersect_with_boundary_pixels' to break the SP along the boundaries
        # (see CREMI-experiments script for an example)
        self.superpixel_gen = WatershedOnDistanceTransformFromAffinities(self.offsets,
                                                                         threshold=ws_threshold,
                                                                         min_segment_size=ws_minsize,
                                                                         preserve_membrane=True,
                                                                         sigma_seeds=ws_sigma,
                                                                         stacked_2d=not ws_2D,
                                                                         used_offsets=[0, 1, 2],
                                                                         offset_weights=[1, 1, 1],
                                                                         n_threads=n_threads)

    def process(self, pmaps):
        # Pmaps are interpreted as affinities
        affinities = np.stack([pmaps, pmaps, pmaps], axis=0)

        # Shift is required to correct aligned affinities
        affinities = shift_affinities(affinities, offsets=self.offsets)

        # invert affinities
        affinities = 1 - affinities

        return self.superpixel_gen(affinities)
