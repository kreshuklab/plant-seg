import os
from concurrent import futures

import h5py
import nifty
import nifty.graph.rag as nrag

import elf.segmentation.lifted_multicut as lmc
from elf.segmentation.features import compute_rag, compute_boundary_mean_and_length, lifted_problem_from_probabilities, \
    project_node_labels_to_pixels, lifted_problem_from_segmentation
from elf.segmentation.multicut import multicut_kernighan_lin, transform_probabilities_to_costs
from elf.segmentation.watershed import distance_transform_watershed

N_THREADS = 16


def segment_volume_mc(pmaps, threshold=0.4, sigma=2.0, beta=0.6, ws=None, sp_min_size=100):
    if ws is None:
        ws = distance_transform_watershed(pmaps, threshold, sigma, min_size=sp_min_size)[0]

    rag = compute_rag(ws, 1)
    features = nrag.accumulateEdgeMeanAndLength(rag, pmaps, numberOfThreads=1)
    probs = features[:, 0]  # mean edge prob
    edge_sizes = features[:, 1]
    costs = transform_probabilities_to_costs(probs, edge_sizes=edge_sizes, beta=beta)
    graph = nifty.graph.undirectedGraph(rag.numberOfNodes)
    graph.insertEdges(rag.uvIds())

    node_labels = multicut_kernighan_lin(graph, costs)

    return nifty.tools.take(node_labels, ws)


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


def read_segment_write(boundary_pmap_files, nuclei_pmap_files=None):
    print(f'Processing: {boundary_pmap_files} and {nuclei_pmap_files}')
    with h5py.File(boundary_pmap_files, 'r') as f:
        boundary_pmaps = f['predictions'][0]

    if nuclei_pmap_files is None:
        print('Running MC...')
        mc = segment_volume_mc(boundary_pmaps)
        output_file = os.path.splitext(boundary_pmap_files)[0] + '_lmc.h5'
    else:
        print('Running LMC...')
        with h5py.File(nuclei_pmap_files, 'r') as f:
            nuclei_pmaps = f['predictions'][0]

        mc = segment_volume_lmc(boundary_pmaps, nuclei_pmaps)
        output_file = os.path.splitext(boundary_pmap_files)[0] + '_mc.h5'

    with h5py.File(output_file, 'w') as f:
        output_dataset = 'segmentation/multicut'
        if output_dataset in f:
            del f[output_dataset]
        print(f'Saving results to {output_file}')
        f.create_dataset(output_dataset, data=mc.astype('uint16'), compression='gzip')


if __name__ == '__main__':
    boundary_pmap_files = [
        '/g/kreshuk/wolny/Datasets/LateralRoot/predictions/unet_ds1x_bce_gcr/Marvelous_t00006_crop_x420_y620_gt_predictions.h5',
        '/g/kreshuk/wolny/Datasets/LateralRoot/predictions/unet_ds1x_bce_gcr/Marvelous_t00045_gt_predictions.h5',
        '/g/kreshuk/wolny/Datasets/LateralRoot/predictions/unet_ds1x_bce_gcr/Beautiful_T00010_crop_x40-1630_y520-970_gt_predictions.h5',
        '/g/kreshuk/wolny/Datasets/LateralRoot/predictions/unet_ds1x_bce_gcr/Beautiful_T00020_crop_x40-1630_y520-970_gt_predictions.h5'
    ]

    nuclei_pmap_files = [
        '/g/kreshuk/wolny/Datasets/LateralRoot/nuclei/Test/Movie1_t00006_nuclei_predictions.h5',
        '/g/kreshuk/wolny/Datasets/LateralRoot/nuclei/Test/Movie1_t00045_nuclei_predictions.h5'
        '/g/kreshuk/wolny/Datasets/LateralRoot/nuclei/Test/Movie2_t00010_nuclei_predictions.h5',
        '/g/kreshuk/wolny/Datasets/LateralRoot/nuclei/Test/Movie2_t00020_nuclei_predictions.h5'

    ]

    with futures.ThreadPoolExecutor(N_THREADS) as tp:
        tasks = []
        for bp, np in zip(boundary_pmap_files, nuclei_pmap_files):
            tasks.append(tp.submit(read_segment_write, bp, np))

        for t in tasks:
            t.result()

    print('Finished!')
