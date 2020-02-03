import numpy as np
import time
from GASP.segmentation import GaspFromAffinities, WatershedOnDistanceTransformFromAffinities
from GASP.segmentation.watershed import SizeThreshAndGrowWithWS
from plantseg import GenericProcessing
from plantseg.segmentation import shift_affinities


class GaspFromPmaps(GenericProcessing):
    def __init__(self,
                 predictions_paths,
                 save_directory="GASP",
                 gasp_linkage_criteria='average',
                 bias=0.5,
                 run_ws=True,
                 ws_2D=True,
                 ws_threshold=0.6,
                 ws_minsize=50,
                 ws_sigma=0.3,
                 post_minsize=50,
                 n_threads=6):

        super().__init__(predictions_paths,
                         input_type="data_float32",
                         output_type="labels",
                         save_directory=save_directory)

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

        # Multithread
        self.n_threads = n_threads

    def __call__(self):
        for predictions_path in self.predictions_paths:
            print(f"Segmenting {predictions_path}")
            output_path, exist = self.create_output_path(predictions_path,
                                                         prefix="_gasp_" + self.gasp_linkage_criteria,
                                                         out_ext=".h5")
            pmaps = self.load_stack(predictions_path)

            # Pmaps are interpreted as affinities
            affinities = np.stack([pmaps, pmaps, pmaps], axis=0)
            # Clean pmaps
            del pmaps

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

            if self.gasp_linkage_criteria == "average" or self.gasp_linkage_criteria == "mutex_watershed":

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

            elif self.gasp_linkage_criteria == "DtWatershed":
                # Watershed segmentation
                segmentation = superpixel_gen(affinities)
            else:
                raise NotImplementedError

            self.save_output(segmentation, output_path, dataset="segmentation")

            # stop real world clock timer
            runtime = time.time() - runtime
            self.runtime = runtime
            print(" - Clustering took {} s".format(runtime))
            self.outputs_paths.append(output_path)

        return self.outputs_paths
