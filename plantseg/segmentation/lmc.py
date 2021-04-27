import os
import time

import elf.segmentation.lifted_multicut as lmc
import numpy as np
from elf.segmentation.features import compute_rag, compute_boundary_mean_and_length
from elf.segmentation.features import lifted_problem_from_probabilities, \
    project_node_labels_to_pixels, lifted_problem_from_segmentation
from elf.segmentation.multicut import transform_probabilities_to_costs
from elf.segmentation.watershed import distance_transform_watershed, apply_size_filter

from plantseg.pipeline import gui_logger
from plantseg.pipeline.steps import AbstractSegmentationStep
from plantseg.pipeline.utils import load_paths


class LiftedMulticut(AbstractSegmentationStep):
    def __init__(self,
                 predictions_paths,
                 nuclei_predictions_path,
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
                 **kwargs):
        super().__init__(input_paths=predictions_paths,
                         save_directory=save_directory,
                         file_suffix='_lmc',
                         state=state)

        self.nuclei_predictions_paths = load_paths(nuclei_predictions_path)

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
        gui_logger.info('Clustering with LiftedMulticut...')
        boundary_pmaps, nuclei_pmaps = pmaps
        runtime = time.time()
        segmentation = segment_volume_lmc(boundary_pmaps, nuclei_pmaps, self.ws_threshold, self.ws_sigma,
                                          self.ws_minsize)

        if self.post_minsize > self.ws_minsize:
            segmentation, _ = apply_size_filter(segmentation, boundary_pmaps, self.post_minsize)

        # stop real world clock timer
        runtime = time.time() - runtime
        gui_logger.info(f"Clustering took {runtime:.2f} s")

        return segmentation

    def read_process_write(self, input_path):
        gui_logger.info(f'Loading stack from {input_path}')
        boundary_pmaps, voxel_size = self.load_stack(input_path)
        nuclei_pmaps_path = self._find_nuclei_pmaps_path(input_path)
        if nuclei_pmaps_path is None:
            raise RuntimeError(f'Cannot find nuclei probability maps for: {input_path}')
        nuclei_pmaps, _ = self.load_stack(nuclei_pmaps_path)

        # pass boundary_pmaps and nuceli_pmaps to process with Lifted Multicut
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
        filename = os.path.split(input_path)[1]
        for nuclei_path in self.nuclei_predictions_paths:
            fn = os.path.split(nuclei_path)[1]
            if fn == filename:
                return nuclei_path
        return None


def segment_volume_lmc(boundary_pmaps, nuclei_pmaps, threshold=0.4, sigma=2.0, sp_min_size=100):
    watershed = distance_transform_watershed(boundary_pmaps, threshold, sigma, min_size=sp_min_size)[0]

    # compute the region adjacency graph
    rag = compute_rag(watershed)

    # compute the edge costs
    features = compute_boundary_mean_and_length(rag, boundary_pmaps)
    costs, sizes = features[:, 0], features[:, 1]

    # transform the edge costs from [0, 1] to  [-inf, inf], which is
    # necessary for the multicut. This is done by intepreting the values
    # as probabilities for an edge being 'true' and then taking the negative log-likelihood.
    # in addition, we weight the costs by the size of the corresponding edge

    # we choose a boundary bias smaller than 0.5 in order to
    # decrease the degree of over segmentation
    boundary_bias = .6

    costs = transform_probabilities_to_costs(costs, edge_sizes=sizes, beta=boundary_bias)
    # compute lifted multicut features from vesicle and dendrite pmaps
    input_maps = [nuclei_pmaps]
    assignment_threshold = .9
    lifted_uvs, lifted_costs = lifted_problem_from_probabilities(rag, watershed,
                                                                 input_maps, assignment_threshold,
                                                                 graph_depth=4)

    # solve the full lifted problem using the kernighan lin approximation introduced in
    # http://openaccess.thecvf.com/content_iccv_2015/html/Keuper_Efficient_Decomposition_of_ICCV_2015_paper.html
    node_labels = lmc.lifted_multicut_kernighan_lin(rag, costs, lifted_uvs, lifted_costs)
    lifted_segmentation = project_node_labels_to_pixels(rag, node_labels)
    return lifted_segmentation


def segment_volume_lmc_from_seg(boundary_pmaps, nuclei_seg, threshold=0.4, sigma=2.0, sp_min_size=100):
    watershed = distance_transform_watershed(boundary_pmaps, threshold, sigma, min_size=sp_min_size)[0]

    # compute the region adjacency graph
    rag = compute_rag(watershed)

    # compute the edge costs
    features = compute_boundary_mean_and_length(rag, boundary_pmaps)
    costs, sizes = features[:, 0], features[:, 1]

    # transform the edge costs from [0, 1] to  [-inf, inf], which is
    # necessary for the multicut. This is done by intepreting the values
    # as probabilities for an edge being 'true' and then taking the negative log-likelihood.
    # in addition, we weight the costs by the size of the corresponding edge

    # we choose a boundary bias smaller than 0.5 in order to
    # decrease the degree of over segmentation
    boundary_bias = .6

    costs = transform_probabilities_to_costs(costs, edge_sizes=sizes, beta=boundary_bias)
    max_cost = np.abs(np.max(costs))
    lifted_uvs, lifted_costs = lifted_problem_from_segmentation(rag, watershed, nuclei_seg, overlap_threshold=0.2,
                                                                graph_depth=4,
                                                                same_segment_cost=5 * max_cost,
                                                                different_segment_cost=-5 * max_cost)

    # solve the full lifted problem using the kernighan lin approximation introduced in
    # http://openaccess.thecvf.com/content_iccv_2015/html/Keuper_Efficient_Decomposition_of_ICCV_2015_paper.html
    node_labels = lmc.lifted_multicut_kernighan_lin(rag, costs, lifted_uvs, lifted_costs)
    lifted_segmentation = project_node_labels_to_pixels(rag, node_labels)
    return lifted_segmentation
