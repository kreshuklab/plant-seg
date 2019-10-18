import h5py
import numpy as np
import os
import time
from GASP.segmentation import GaspFromAffinities, WatershedOnDistanceTransformFromAffinities
from GASP.segmentation.watershed import SizeThreshAndGrowWithWS


class GaspFromPmaps:
    def __init__(self,
                 save_directory="GASP",
                 gasp_linkage_criteria='average',
                 gasp_beta_bias=0.5,
                 run_ws=True,
                 ws_threshold=0.6,
                 ws_minsize=50,
                 ws_sigma=0.3,
                 post_minsize=50,
                 n_threads=6):

        # name subdirectory created for the segmentation file + generic config
        self.save_directory = save_directory
        self.n_threads = n_threads

        # GASP parameters
        self.gasp_linkage_criteria = gasp_linkage_criteria
        self.gasp_beta_bias = gasp_beta_bias

        # Watershed parameters
        self.run_ws = run_ws
        self.ws_threshold = ws_threshold
        self.ws_minsize = ws_minsize
        self.ws_sigma = ws_sigma

        # Post processing size threshold
        self.post_minsize = post_minsize

    def __call__(self, predictions_path):

        # Generate some random affinities:
        affinities = h5py.File(predictions_path, "r")
        affinities = np.array(affinities["predictions"], dtype=np.float32)

        # Pmaps are interpreted as affinities
        affinities = np.concatenate((affinities[[0]],
                                     affinities[[0]],
                                     affinities[[0]]))

        offsets = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
        # Shift is required to correct alligned affinities
        affinities = self.shift_affinities(affinities, offsets=offsets)

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
                                                                        stacked_2d=False,
                                                                        used_offsets=[0, 1, 2],
                                                                        offset_weights=[0, 1, 2],
                                                                        n_threads=self.n_threads)

        else:
            superpixel_gen = None

        run_GASP_kwargs = {'linkage_criteria': self.gasp_linkage_criteria,
                           'add_cannot_link_constraints': False,
                           'use_efficient_implementations': False}

        # Init and run Gasp
        gasp_instance = GaspFromAffinities(offsets,
                                           superpixel_generator=superpixel_gen,
                                           run_GASP_kwargs=run_GASP_kwargs,
                                           n_threads=self.n_threads,
                                           beta_bias=self.gasp_beta_bias)

        runtime = time.time()
        final_segmentation, _ = gasp_instance(affinities)

        # init and run size threshold
        size_tresh = SizeThreshAndGrowWithWS(self.post_minsize, offsets)
        final_segmentation = size_tresh(affinities, final_segmentation)

        runtime = time.time() - runtime

        os.makedirs(os.path.dirname(predictions_path) + "/" + self.save_directory + "/", exist_ok=True)
        h5_file_path = (os.path.dirname(predictions_path) +
                        "/" + self.save_directory + "/" + os.path.basename(predictions_path))

        h5_file_path = os.path.splitext(h5_file_path)[0] + "_gasp_" + run_GASP_kwargs['linkage_criteria'] + ".h5"

        self.runtime = runtime
        self._log_params(h5_file_path)
        # Save output results
        with h5py.File(h5_file_path, "w") as file:
            file.create_dataset("segmentation", data=final_segmentation.astype(np.uint16), compression='gzip')

        print("Clustering took {} s".format(runtime))
        return h5_file_path

    def _log_params(self, file):
        import yaml
        file = os.path.splitext(file)[0] + ".yaml"
        dict_file = {"algorithm": self.__class__.__name__}

        for name, value in self.__dict__.items():
            dict_file[name] = value

        with open(file, "w") as f:
            f.write(yaml.dump(dict_file))

    @staticmethod
    def shift_affinities(affinities, offsets):
        rolled_affs = []
        for i, _ in enumerate(offsets):
            offset = offsets[i]
            shifts = tuple([int(off / 2) for off in offset])

            padding = [[0, 0] for _ in range(len(shifts))]
            for ax, shf in enumerate(shifts):
                if shf < 0:
                    padding[ax][1] = -shf
                elif shf > 0:
                    padding[ax][0] = shf
            padded_inverted_affs = np.pad(affinities, pad_width=((0, 0),) + tuple(padding), mode='constant')
            crop_slices = tuple(
                slice(padding[ax][0], padded_inverted_affs.shape[ax + 1] - padding[ax][1]) for ax in range(3))
            rolled_affs.append(np.roll(padded_inverted_affs[i], shifts, axis=(0, 1, 2))[crop_slices])

        return np.stack(rolled_affs)
