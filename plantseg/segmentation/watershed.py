import h5py
import numpy as np
import os
from GASP.segmentation import WatershedOnDistanceTransformFromAffinities
from GASP.segmentation.watershed import SizeThreshAndGrowWithWS
import time
import tifffile


class DtWatershedFromPmaps:
    def __init__(self,
                 predictions_paths,
                 save_directory="Watershed",
                 run_ws=True,
                 ws_threshold=0.6,
                 ws_minsize=50,
                 ws_sigma=0.3,
                 post_minsize=50,
                 n_threads=6):

        # name subdirectory created for the segmentation file + generic config
        self.predictions_paths = predictions_paths
        self.save_directory = save_directory
        self.n_threads = n_threads

        # Watershed parameters
        self.run_ws = run_ws
        self.ws_threshold = ws_threshold
        self.ws_minsize = ws_minsize
        self.ws_sigma = ws_sigma

        # Post processing size threshold
        self.post_minsize = post_minsize

    def __call__(self):
        for predictions_path in self.predictions_paths:
            # Load file
            _, ext = os.path.splitext(predictions_path)
            pmaps = None
            if ext == ".tiff" or ext == ".tif":
                pmaps = tifffile.imread(predictions_path)

                # squeeze extra dimension
                if len(pmaps.shape) == 4:
                    pmaps = pmaps[0]

                pmaps = (pmaps - pmaps.min()) / (pmaps.max() - pmaps.min()).astype(np.float32)

            elif ext == ".hdf" or ext == ".h5" or ext == ".hd5":
                with h5py.File(predictions_path, "r") as f:
                    # Check for h5 dataset
                    if "predictions" in f.keys():
                        # predictions is the first choice
                        dataset = "predictions"
                    elif "raw" in f.keys():
                        # raw is the second choice
                        dataset = "raw"
                    else:
                        print("H5 dataset name not understood")
                        raise NotImplementedError

                    # Load data
                    if len(f[dataset].shape) == 3:
                        pmaps = f[dataset][...].astype(np.float32)
                    elif len(f[dataset].shape) == 4:
                        pmaps = f[dataset][0, ...].astype(np.float32)
                    else:
                        print(f[dataset].shape)
                        print("Data shape not understood, data must be 3D or 4D")
                        raise NotImplementedError

            else:
                print("Data extension not understood")
                raise NotImplementedError
            assert pmaps.ndim == 3, "Input probability maps must be 3D tiff or h5 (zxy) or" \
                                    " 4D (czxy)," \
                                    " where the fist channel contains the neural network boundary predictions"

            # Pmaps are interpreted as affinities
            affinities = np.stack([pmaps, pmaps, pmaps], axis=0)

            offsets = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
            # Shift is required to correct alligned affinities
            affinities = self.shift_affinities(affinities, offsets=offsets)

            # invert affinities
            affinities = 1 - affinities

            watershed_seg = WatershedOnDistanceTransformFromAffinities(offsets,
                                                                       threshold=self.ws_threshold,
                                                                       min_segment_size=self.ws_minsize,
                                                                       preserve_membrane=True,
                                                                       sigma_seeds=self.ws_sigma,
                                                                       stacked_2d=False,
                                                                       used_offsets=[0, 1, 2],
                                                                       offset_weights=[0, 1, 2],
                                                                       n_threads=self.n_threads)

            runtime = time.time()
            final_segmentation = watershed_seg(affinities)

            # init and run size threshold
            size_tresh = SizeThreshAndGrowWithWS(self.post_minsize, offsets)
            final_segmentation = size_tresh(affinities, final_segmentation)
            runtime = time.time() - runtime

            os.makedirs(os.path.join(os.path.dirname(predictions_path),
                                     self.save_directory), exist_ok=True)

            h5_file_path = os.path.join(os.path.dirname(predictions_path),
                                        self.save_directory,
                                        os.path.basename(predictions_path))

            h5_file_path = os.path.splitext(h5_file_path)[0] + "_watershed" + ".h5"

            self.runtime = runtime
            self._log_params(h5_file_path)

            # Save output results
            with h5py.File(h5_file_path, "w") as file:
                file.create_dataset("segmentation", data=final_segmentation.astype(np.uint16), compression='gzip')
            print("Clustering took {} s".format(runtime))

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