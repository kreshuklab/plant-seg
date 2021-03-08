import time

import nifty
from elf.segmentation.features import compute_rag, compute_boundary_mean_and_length
from elf.segmentation.multicut import multicut_kernighan_lin, transform_probabilities_to_costs
from elf.segmentation.watershed import distance_transform_watershed, apply_size_filter, stacked_watershed

from plantseg.pipeline import gui_logger
from plantseg.pipeline.steps import AbstractSegmentationStep


class MulticutFromPmaps(AbstractSegmentationStep):
    def __init__(self,
                 predictions_paths,
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
                 **kwargs):

        super().__init__(input_paths=predictions_paths,
                         save_directory=save_directory,
                         file_suffix='_multicut',
                         state=state)

        self.beta = beta

        # Watershed parameters
        self.run_ws = run_ws
        self.ws_2D = ws_2D
        self.ws_threshold = ws_threshold
        self.ws_minsize = ws_minsize
        self.ws_sigma = ws_sigma
        self.ws_w_sigma = ws_w_sigma

        # Post processing size threshold
        self.post_minsize = post_minsize

        # Multithread
        self.n_threads = n_threads

    def process(self, pmaps):
        gui_logger.info('Clustering with MultiCut...')
        runtime = time.time()
        segmentation = self.segment_volume(pmaps)
        segmentation = segmentation.astype('uint32')

        if self.post_minsize > self.ws_minsize:
            gui_logger.info("Applying size filter for post-processing")
            segmentation, _ = apply_size_filter(segmentation, pmaps, self.post_minsize)

        # stop real world clock timer
        runtime = time.time() - runtime
        gui_logger.info(f"Clustering took {runtime:.2f} s")

        return segmentation

    def segment_volume(self, pmaps):
        if self.ws_2D:
            # WS in 2D
            gui_logger.info("Computing watershed in 2d")
            ws, n_labels = stacked_watershed(pmaps, n_threads=self.n_threads,
                                             threshold=self.ws_threshold,
                                             sigma_seeds=self.ws_sigma,
                                             sigma_weights=self.ws_w_sigma,
                                             min_size=self.ws_minsize)
        else:
            # WS in 3D
            gui_logger.info("Computing watershed in 3d")
            ws, n_labels = distance_transform_watershed(pmaps, self.ws_threshold,
                                                        self.ws_sigma,
                                                        sigma_weights=self.ws_w_sigma,
                                                        min_size=self.ws_minsize)

        gui_logger.info("Computing region adjacency graph")
        n_labels += 1
        rag = compute_rag(ws, n_labels=n_labels, n_threads=self.n_threads)

        gui_logger.info("Computing edge_features")
        features = compute_boundary_mean_and_length(rag, pmaps, n_threads=self.n_threads)
        # Computing edge features
        probs = features[:, 0]  # mean edge prob
        edge_sizes = features[:, 1]
        # Prob -> edge costs
        costs = transform_probabilities_to_costs(probs, edge_sizes=edge_sizes, beta=self.beta)

        # Solving Multicut
        gui_logger.info("Computing multicut")
        node_labels = multicut_kernighan_lin(rag, costs)
        return nifty.tools.take(node_labels, ws)
