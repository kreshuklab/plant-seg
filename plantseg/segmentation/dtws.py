import numpy as np
from functools import partial
from plantseg.pipeline.steps import AbstractSegmentationStep
from elf.segmentation.watershed import distance_transform_watershed, stacked_watershed


def compute_distance_transfrom_watershed(pmaps, threshold, sigma_seeds,
                                         stacked=False, sigma_weights=2.,
                                         min_size=100, alpha=.9,
                                         pixel_pitch=None, apply_nonmax_suppression=False,
                                         n_threads=None):
    """ Wrapper around elf.distance_transform_watershed

    Args:
        pmaps (np.ndarray): input height map.
        threshold (float): value for the threshold applied before distance transform.
        sigma_seeds (float): smoothing factor for the watershed seed map.
        stacked (bool): if true the ws will be executed in 2D slice by slice, otherwise in 3D.
        sigma_weights (float): smoothing factor for the watershed weight map (default: 2).
        min_size (int): minimal size of watershed segments (default: 100)
        alpha (float): alpha used to blend input_ and distance_transform in order to obtain the
            watershed weight map (default: .9)
        pixel_pitch (list-like[int]): anisotropy factor used to compute the distance transform (default: None)
        apply_nonmax_suppression (bool): whether to apply non-maximum suppression to filter out seeds.
            Needs nifty. (default: False)
        n_threads (int): if not None, parallelize the 2D stacked ws. (default: None)

    Returns:
        np.ndarray: watershed segmentation
    """

    ws_kwargs = dict(threshold=threshold, sigma_seeds=sigma_seeds,
                     sigma_weights=sigma_weights,
                     min_size=min_size, alpha=alpha,
                     pixel_pitch=pixel_pitch, apply_nonmax_suppression=apply_nonmax_suppression)
    if stacked:
        # WS in 2D
        ws, _ = stacked_watershed(pmaps, ws_function=distance_transform_watershed, n_threads=n_threads, **ws_kwargs)
    else:
        # WS in 3D
        ws, _ = distance_transform_watershed(pmaps, **ws_kwargs)

    return ws


class DistanceTransformWatershed(AbstractSegmentationStep):
    def __init__(self,
                 predictions_paths,
                 save_directory="DTWatershed",
                 ws_2D=True,
                 ws_threshold=0.4,
                 ws_minsize=50,
                 ws_sigma=0.3,
                 ws_w_sigma=0,
                 n_threads=8,
                 state=True,
                 **kwargs):
        super().__init__(input_paths=predictions_paths,
                         save_directory=save_directory,
                         file_suffix='_dtws',
                         state=state)

        self.dt_watershed = partial(compute_distance_transfrom_watershed,
                                    threshold=ws_threshold, sigma_seeds=ws_sigma,
                                    stacked=ws_2D, sigma_weights=ws_w_sigma,
                                    min_size=ws_minsize, n_threads=n_threads)

    def process(self, pmaps):
        return self.dt_watershed(pmaps)
