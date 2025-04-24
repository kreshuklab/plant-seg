import os
from concurrent import futures

import h5py
import nifty
import nifty.graph.rag as nrag
from elf.segmentation.features import compute_rag
from elf.segmentation.multicut import (
    multicut_kernighan_lin,
    transform_probabilities_to_costs,
)
from elf.segmentation.watershed import distance_transform_watershed

from plantseg.segmentation.functional.segmentation import (
    lifted_multicut_from_nuclei_pmaps,
)

N_THREADS = 16


def segment_volume_mc(
    pmaps, threshold=0.4, sigma=2.0, beta=0.6, ws=None, sp_min_size=100
):
    if ws is None:
        ws = distance_transform_watershed(
            pmaps, threshold, sigma, min_size=sp_min_size
        )[0]

    rag = compute_rag(ws, 1)
    features = nrag.accumulateEdgeMeanAndLength(rag, pmaps, numberOfThreads=1)
    probs = features[:, 0]  # mean edge prob
    edge_sizes = features[:, 1]
    costs = transform_probabilities_to_costs(probs, edge_sizes=edge_sizes, beta=beta)
    graph = nifty.graph.undirectedGraph(rag.numberOfNodes)
    graph.insertEdges(rag.uvIds())

    node_labels = multicut_kernighan_lin(graph, costs)

    return nifty.tools.take(node_labels, ws)


def read_segment_write(boundary_pmap_files, nuclei_pmap_files=None):
    print(f"Processing: {boundary_pmap_files} and {nuclei_pmap_files}")
    with h5py.File(boundary_pmap_files, "r") as f:
        boundary_pmaps = f["predictions"][0]

    if nuclei_pmap_files is None:
        print("Running MC...")
        mc = segment_volume_mc(boundary_pmaps)
        output_file = os.path.splitext(boundary_pmap_files)[0] + "_lmc.h5"
    else:
        print("Running LMC...")
        with h5py.File(nuclei_pmap_files, "r") as f:
            nuclei_pmaps = f["predictions"][0]

        mc = lifted_multicut_from_nuclei_pmaps(boundary_pmaps, nuclei_pmaps)
        output_file = os.path.splitext(boundary_pmap_files)[0] + "_mc.h5"

    with h5py.File(output_file, "w") as f:
        output_dataset = "segmentation/multicut"
        if output_dataset in f:
            del f[output_dataset]
        print(f"Saving results to {output_file}")
        f.create_dataset(output_dataset, data=mc.astype("uint16"), compression="gzip")


if __name__ == "__main__":
    boundary_pmap_files = [
        "/g/kreshuk/wolny/Datasets/LateralRoot/predictions/unet_ds1x_bce_gcr/Marvelous_t00006_crop_x420_y620_gt_predictions.h5",
        "/g/kreshuk/wolny/Datasets/LateralRoot/predictions/unet_ds1x_bce_gcr/Marvelous_t00045_gt_predictions.h5",
        "/g/kreshuk/wolny/Datasets/LateralRoot/predictions/unet_ds1x_bce_gcr/Beautiful_T00010_crop_x40-1630_y520-970_gt_predictions.h5",
        "/g/kreshuk/wolny/Datasets/LateralRoot/predictions/unet_ds1x_bce_gcr/Beautiful_T00020_crop_x40-1630_y520-970_gt_predictions.h5",
    ]

    nuclei_pmap_files = [
        "/g/kreshuk/wolny/Datasets/LateralRoot/nuclei/Test/Movie1_t00006_nuclei_predictions.h5",
        "/g/kreshuk/wolny/Datasets/LateralRoot/nuclei/Test/Movie1_t00045_nuclei_predictions.h5"
        "/g/kreshuk/wolny/Datasets/LateralRoot/nuclei/Test/Movie2_t00010_nuclei_predictions.h5",
        "/g/kreshuk/wolny/Datasets/LateralRoot/nuclei/Test/Movie2_t00020_nuclei_predictions.h5",
    ]

    with futures.ThreadPoolExecutor(N_THREADS) as tp:
        tasks = []
        for bp, np in zip(boundary_pmap_files, nuclei_pmap_files):
            tasks.append(tp.submit(read_segment_write, bp, np))

        for t in tasks:
            t.result()

    print("Finished!")
