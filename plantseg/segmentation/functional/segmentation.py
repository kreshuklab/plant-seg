import nifty
import nifty.graph.rag as nrag
import numpy as np
from elf.segmentation import GaspFromAffinities
from elf.segmentation import stacked_watershed, lifted_multicut as lmc, \
    project_node_labels_to_pixels
from elf.segmentation.features import compute_rag, lifted_problem_from_probabilities, lifted_problem_from_segmentation
from elf.segmentation.multicut import multicut_kernighan_lin
from elf.segmentation.watershed import distance_transform_watershed, apply_size_filter
from vigra.filters import gaussianSmoothing

from plantseg.segmentation.functional.utils import shift_affinities, compute_mc_costs

try:
    import SimpleITK as sitk

    sitk_installed = True
except ImportError:
    sitk_installed = False


def dt_watershed(boundary_pmaps: np.array,
                 threshold: float = 0.5,
                 sigma_seeds: float = 1.,
                 stacked: bool = False,
                 sigma_weights: float = 2.,
                 min_size: int = 100,
                 alpha: float = 1.0,
                 pixel_pitch: tuple[int, ...] = None,
                 apply_nonmax_suppression: bool = False,
                 n_threads: int = None,
                 mask: np.array = None) -> np.array:
    """ Wrapper around elf.distance_transform_watershed

    Args:
        boundary_pmaps (np.ndarray): input height map.
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
        mask (np.ndarray)

    Returns:
        np.ndarray: watershed segmentation
    """

    boundary_pmaps = boundary_pmaps.astype('float32')
    ws_kwargs = dict(threshold=threshold, sigma_seeds=sigma_seeds,
                     sigma_weights=sigma_weights,
                     min_size=min_size, alpha=alpha,
                     pixel_pitch=pixel_pitch,
                     apply_nonmax_suppression=apply_nonmax_suppression,
                     mask=mask)
    if stacked:
        # WS in 2D
        ws, _ = stacked_watershed(boundary_pmaps,
                                  ws_function=distance_transform_watershed,
                                  n_threads=n_threads,
                                  **ws_kwargs)
    else:
        # WS in 3D
        ws, _ = distance_transform_watershed(boundary_pmaps, **ws_kwargs)

    return ws


def gasp(boundary_pmaps: np.array,
         superpixels: np.array = None,
         gasp_linkage_criteria: str = 'average',
         beta: float = 0.5,
         post_minsize: int = 100,
         n_threads: int = 6) -> np.array:
    """
    Implementation of the GASP algorithm for segmentation from affinities.

    Args:
        boundary_pmaps (np.ndarray): cell boundary predictions.
        superpixels (np.ndarray): superpixel segmentation. If None, GASP will be run from the pixels. (default: None)
        gasp_linkage_criteria (str): Linkage criteria for GASP. (default: 'average')
        beta (float): beta parameter for GASP. A small value will steer the segmentation towards under-segmentation.
        While a high-value bias the segmentation towards the over-segmentation. (default: 0.5)
        post_minsize (int): minimal size of the segments after GASP. (default: 100)
        n_threads (int): number of threads used for GASP. (default: 6)

    Returns:
        np.ndarray: GASP output segmentation

    """
    if superpixels is not None:
        assert boundary_pmaps.shape == superpixels.shape

        if superpixels.ndim == 2:
            superpixels = superpixels[None, ...]

        def superpixel_gen(*args, **kwargs):
            return superpixels
    else:
        superpixel_gen = None

    if boundary_pmaps.ndim == 2:
        boundary_pmaps = boundary_pmaps[None, ...]

    run_GASP_kwargs = {'linkage_criteria': gasp_linkage_criteria,
                       'add_cannot_link_constraints': False,
                       'use_efficient_implementations': False}

    # pmaps are interpreted as affinities
    boundary_pmaps = boundary_pmaps.astype('float32')
    affinities = np.stack([boundary_pmaps, boundary_pmaps, boundary_pmaps], axis=0)

    offsets = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    # Shift is required to correct aligned affinities
    affinities = shift_affinities(affinities, offsets=offsets)

    # invert affinities
    affinities = 1 - affinities

    # Init and run Gasp
    gasp_instance = GaspFromAffinities(offsets,
                                       superpixel_generator=superpixel_gen,
                                       run_GASP_kwargs=run_GASP_kwargs,
                                       n_threads=n_threads,
                                       beta_bias=beta)
    # running gasp
    segmentation, _ = gasp_instance(affinities)

    # init and run size threshold
    if post_minsize > 0:
        segmentation, _ = apply_size_filter(segmentation.astype('uint32'), boundary_pmaps, post_minsize)
    return segmentation


def mutex_ws(boundary_pmaps: np.array,
             superpixels: np.array = None,
             beta: float = 0.5,
             post_minsize: int = 100,
             n_threads: int = 6) -> np.array:
    """
    Wrapper around gasp with mutex_watershed as linkage criteria.

    Args:
        boundary_pmaps (np.ndarray): cell boundary predictions. 3D array of shape (Z, Y, X) with values between 0 and 1.
        superpixels (np.ndarray): superpixel segmentation. Must have the same shape as boundary_pmaps.
            If None, GASP will be run from the pixels. (default: None)
        beta (float): beta parameter for GASP. A small value will steer the segmentation towards under-segmentation.
            While a high-value bias the segmentation towards the over-segmentation. (default: 0.5)
        post_minsize (int): minimal size of the segments after GASP. (default: 100)
        n_threads (int): number of threads used for GASP. (default: 6)

    Returns:
        np.ndarray: GASP output segmentation

    """
    return gasp(boundary_pmaps=boundary_pmaps,
                superpixels=superpixels,
                gasp_linkage_criteria='mutex_watershed',
                beta=beta,
                post_minsize=post_minsize,
                n_threads=n_threads)


def multicut(boundary_pmaps: np.array,
             superpixels: np.array,
             beta: float = 0.5,
             post_minsize: int = 50) -> np.array:

    """
    Multicut segmentation from boundary predictions.

    Args:
        boundary_pmaps (np.ndarray): cell boundary predictions, 3D array of shape (Z, Y, X) with values between 0 and 1.
        superpixels (np.ndarray): superpixel segmentation. Must have the same shape as boundary_pmaps.
        beta (float): beta parameter for the Multicut. A small value will steer the segmentation towards
            under-segmentation. While a high-value bias the segmentation towards the over-segmentation. (default: 0.5)
        post_minsize (int): minimal size of the segments after Multicut. (default: 100)

    Returns:
        np.ndarray: Multicut output segmentation
    """

    rag = compute_rag(superpixels)

    # Prob -> edge costs
    boundary_pmaps = boundary_pmaps.astype('float32')
    costs = compute_mc_costs(boundary_pmaps, rag, beta=beta)

    # Creating graph
    graph = nifty.graph.undirectedGraph(rag.numberOfNodes)
    graph.insertEdges(rag.uvIds())

    # Solving Multicut
    node_labels = multicut_kernighan_lin(graph, costs)
    segmentation = nifty.tools.take(node_labels, superpixels)

    # run size threshold
    if post_minsize > 0:
        segmentation, _ = apply_size_filter(segmentation.astype('uint32'),
                                            boundary_pmaps,
                                            post_minsize)
    return segmentation


def lifted_multicut_from_nuclei_pmaps(boundary_pmaps: np.array,
                                      nuclei_pmaps: np.array,
                                      superpixels: np.array,
                                      beta: float = 0.5,
                                      post_minsize: int = 50) -> np.array:
    """
    Lifted Multicut segmentation from boundary predictions and nuclei predictions.

    Args:
        boundary_pmaps (np.ndarray): cell boundary predictions, 3D array of shape (Z, Y, X) with values between 0 and 1.
        nuclei_pmaps (np.array): nuclei predictions. Must have the same shape as boundary_pmaps and
            with values between 0 and 1.
        superpixels (np.ndarray): superpixel segmentation. Must have the same shape as boundary_pmaps.
        beta (float): beta parameter for the Multicut. A small value will steer the segmentation towards
        under-segmentation. While a high-value bias the segmentation towards the over-segmentation. (default: 0.5)
        post_minsize (int): minimal size of the segments after Multicut. (default: 100)

    Returns:
        np.ndarray: Multicut output segmentation
    """
    # compute the region adjacency graph
    rag = compute_rag(superpixels)

    # compute multi cut edges costs
    boundary_pmaps = boundary_pmaps.astype('float32')
    costs = compute_mc_costs(boundary_pmaps, rag, beta)

    # assert nuclei pmaps are floats
    nuclei_pmaps = nuclei_pmaps.astype('float32')
    input_maps = [nuclei_pmaps]
    assignment_threshold = .9

    # compute lifted multicut features from boundary pmaps
    lifted_uvs, lifted_costs = lifted_problem_from_probabilities(rag, superpixels,
                                                                 input_maps, assignment_threshold,
                                                                 graph_depth=4)

    # solve the full lifted problem using the kernighan lin approximation introduced in
    # http://openaccess.thecvf.com/content_iccv_2015/html/Keuper_Efficient_Decomposition_of_ICCV_2015_paper.html
    node_labels = lmc.lifted_multicut_kernighan_lin(rag, costs, lifted_uvs, lifted_costs)
    segmentation = project_node_labels_to_pixels(rag, node_labels)

    # run size threshold
    if post_minsize > 0:
        segmentation, _ = apply_size_filter(segmentation.astype('uint32'), boundary_pmaps, post_minsize)
    return segmentation


def lifted_multicut_from_nuclei_segmentation(boundary_pmaps: np.array,
                                             nuclei_seg: np.array,
                                             superpixels: np.array,
                                             beta: float = 0.5,
                                             post_minsize: int = 50) -> np.array:
    """
    Lifted Multicut segmentation from boundary predictions and nuclei segmentation.

    Args:
        boundary_pmaps (np.ndarray): cell boundary predictions, 3D array of shape (Z, Y, X) with values between 0 and 1.
        nuclei_seg (np.array): Nuclei segmentation. Must have the same shape as boundary_pmaps.
        superpixels (np.ndarray): superpixel segmentation. Must have the same shape as boundary_pmaps.
        beta (float): beta parameter for the Multicut. A small value will steer the segmentation towards
        under-segmentation. While a high-value bias the segmentation towards the over-segmentation. (default: 0.5)
        post_minsize (int): minimal size of the segments after Multicut. (default: 100)

    Returns:
        np.ndarray: Multicut output segmentation
    """
    # compute the region adjacency graph
    rag = compute_rag(superpixels)

    # compute multi cut edges costs
    boundary_pmaps = boundary_pmaps.astype('float32')
    costs = compute_mc_costs(boundary_pmaps, rag, beta)
    max_cost = np.abs(np.max(costs))
    lifted_uvs, lifted_costs = lifted_problem_from_segmentation(rag, superpixels, nuclei_seg,
                                                                overlap_threshold=0.2,
                                                                graph_depth=4,
                                                                same_segment_cost=5 * max_cost,
                                                                different_segment_cost=-5 * max_cost)

    # solve the full lifted problem using the kernighan lin approximation introduced in
    # http://openaccess.thecvf.com/content_iccv_2015/html/Keuper_Efficient_Decomposition_of_ICCV_2015_paper.html
    lifted_costs = lifted_costs.astype('float64')
    node_labels = lmc.lifted_multicut_kernighan_lin(rag, costs, lifted_uvs, lifted_costs)
    segmentation = project_node_labels_to_pixels(rag, node_labels)

    # run size threshold
    if post_minsize > 0:
        segmentation, _ = apply_size_filter(segmentation.astype('uint32'), boundary_pmaps, post_minsize)
    return segmentation


def simple_itk_watershed(boundary_pmaps: np.array,
                         threshold: float = 0.5,
                         sigma: float = 1.0,
                         minsize: int = 100) -> np.array:
    """
    Simple itk watershed segmentation.

    Args:
        boundary_pmaps (np.ndarray): cell boundary predictions. 3D array of shape (Z, Y, X) with values between 0 and 1.
        threshold (float): threshold for the watershed segmentation. (default: 0.5)
        sigma (float): sigma for the gaussian smoothing. (default: 1.0)
        minsize (int): minimal size of the segments after segmentation. (default: 100)

    Returns:
        np.ndarray: simple itk output segmentation

    """
    if not sitk_installed:
        raise ValueError('please install sitk before running this process')

    if sigma > 0:
        # fix ws sigma length
        # ws sigma cannot be shorter than pmaps dims
        max_sigma = (np.array(boundary_pmaps.shape) - 1) / 3
        ws_sigma = np.minimum(max_sigma, np.ones(max_sigma.ndim) * sigma)
        boundary_pmaps = gaussianSmoothing(boundary_pmaps, ws_sigma)

    # Itk watershed + size filtering
    itk_pmaps = sitk.GetImageFromArray(boundary_pmaps)
    itk_segmentation = sitk.MorphologicalWatershed(itk_pmaps,
                                                   threshold,
                                                   markWatershedLine=False,
                                                   fullyConnected=False)
    itk_segmentation = sitk.RelabelComponent(itk_segmentation, minsize)
    segmentation = sitk.GetArrayFromImage(itk_segmentation).astype(np.uint16)
    return segmentation


def simple_itk_watershed_from_markers(boundary_pmaps: np.array,
                                      seeds: np.array):
    if not sitk_installed:
        raise ValueError('please install sitk before running this process')

    itk_pmaps = sitk.GetImageFromArray(boundary_pmaps)
    itk_seeds = sitk.GetImageFromArray(seeds)
    segmentation = sitk.MorphologicalWatershedFromMarkers(itk_pmaps, itk_seeds, markWatershedLine=False,
                                                          fullyConnected=False)
    return sitk.GetArrayFromImage(segmentation).astype('uint32')
