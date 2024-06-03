import os
import time
from functools import partial

from plantseg.pipeline import gui_logger
from plantseg.pipeline.steps import AbstractSegmentationStep
from plantseg.pipeline.utils import load_paths
from plantseg.segmentation.functional.segmentation import (
    dt_watershed,
    lifted_multicut_from_nuclei_pmaps,
    lifted_multicut_from_nuclei_segmentation,
)


class LiftedMulticut(AbstractSegmentationStep):
    def __init__(
        self,
        predictions_paths,
        nuclei_predictions_path,
        key=None,
        key_nuclei=None,
        channel=None,
        channel_nuclei=None,
        is_segmentation=False,
        save_directory="LiftedMulticut",
        beta=0.6,
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
            file_suffix='_lmc',
            state=state,
            input_key=key,
            input_channel=channel,
        )

        self.nuclei_predictions_paths = load_paths(nuclei_predictions_path)
        self.key_nuclei = key_nuclei
        self.channel_nuclei = channel_nuclei
        self.is_segmentation = is_segmentation

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
        boundary_pmaps, nuclei_pmaps = pmaps

        runtime = time.time()
        gui_logger.info('Computing segmentation with dtWS...')
        ws = self.dt_watershed(boundary_pmaps)

        gui_logger.info('Clustering with LiftedMulticut...')
        if self.is_segmentation:
            segmentation = lifted_multicut_from_nuclei_segmentation(
                boundary_pmaps, nuclei_pmaps, superpixels=ws, beta=self.beta, post_minsize=self.post_minsize
            )
        else:
            segmentation = lifted_multicut_from_nuclei_pmaps(
                boundary_pmaps, nuclei_pmaps, superpixels=ws, beta=self.beta, post_minsize=self.post_minsize
            )
        # stop real world clock timer
        runtime = time.time() - runtime
        gui_logger.info(f"Clustering took {runtime:.2f} s")

        return segmentation

    def read_process_write(self, input_path):
        gui_logger.info(f'Loading stack from {input_path}')
        boundary_pmaps, voxel_size = self.load_stack(input_path)

        self.input_key = self.key_nuclei  # load_stack() uses self.input_key to load dataset
        self.input_channel = self.channel_nuclei  # load_stack() uses self.input_channel to select channel
        nuclei_pmaps_path = self._find_nuclei_pmaps_path(input_path)
        if nuclei_pmaps_path is None:
            raise RuntimeError(
                f'Cannot find nuclei probability maps for: {input_path}. '
                f'Nuclei files: {self.nuclei_predictions_paths}'
            )
        nuclei_pmaps, _ = self.load_stack(nuclei_pmaps_path, check_input_type=False)

        # pass boundary_pmaps and nuclei_pmaps to process with Lifted Multicut
        pmaps = (boundary_pmaps, nuclei_pmaps)
        output_data = self.process(pmaps)

        output_path = self._create_output_path(input_path)
        gui_logger.info(f'Saving results in {output_path}')
        self.save_output(output_data, output_path, voxel_size)

        if self.save_raw:
            self.save_raw_dataset(input_path, output_path, voxel_size)

        # return output_path
        return output_path

    def _find_nuclei_pmaps_path(self, input_path):
        if len(self.nuclei_predictions_paths) == 1:
            return self.nuclei_predictions_paths[0]

        # if more than one nuclei pmaps file given in the config, match by filename
        filename = os.path.split(input_path)[1]
        for nuclei_path in self.nuclei_predictions_paths:
            fn = os.path.split(nuclei_path)[1]
            if fn == filename:
                return nuclei_path
        return None
