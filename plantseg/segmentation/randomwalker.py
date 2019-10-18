import h5py
import numpy as np
import os
import time
import vigra
import vigra.filters as ff
from randomwalkerstools.randomwalker_algorithm import random_walker_algorithm_3d, random_walker_algorithm_2d
from elf.segmentation.watershed import apply_size_filter


import os
import glob
import numpy as np
import time
import h5py
import nifty
import nifty.graph.rag as nrag
from elf.segmentation.watershed import apply_size_filter
from elf.segmentation.features import compute_rag
from elf.segmentation.multicut import multicut_kernighan_lin, transform_probabilities_to_costs
from skimage.segmentation import random_walker
from scipy.ndimage import zoom


def segment_volume(pmaps, os):
    rag = compute_rag(os, 1)
    # Computing edge features
    features = nrag.accumulateEdgeMeanAndLength(rag, pmaps, numberOfThreads=1)  # DO NOT CHANGE numberOfThreads
    probs = features[:, 0]  # mean edge prob
    edge_sizes = features[:, 1]
    # Prob -> edge costs
    costs = transform_probabilities_to_costs(probs, edge_sizes=edge_sizes, beta=0.5)
    # Creating graph
    graph = nifty.graph.undirectedGraph(rag.numberOfNodes)
    graph.insertEdges(rag.uvIds())
    # Solving multicut
    node_labels = multicut_kernighan_lin(graph, costs)
    return nifty.tools.take(node_labels, os)


class DtRandomWalkerFromPmaps:
    def __init__(self,
                 save_directory="RandomWalker",
                 rw_type="2D",  # 2D, 3D
                 rw_beta=900,
                 rw_threshold=0.4,
                 rw_minsize=50,
                 rw_sigma=1.0,
                 post_minsize=50,
                 n_threads=6):

        # name subdirectory created for the segmentation file + generic config
        self.save_directory = save_directory
        self.n_threads = n_threads

        # Watershed parameters
        if rw_type == "3D":
            self.rw = self.dt_randomwalker
        elif rw_type == "2D":
            self.rw = self.dt_rw_2d
        elif rw_type == "2D_blocks":
            self.rw = self.dt_rw_2d_blocks
        else:
            raise NotImplementedError

        self.rw_beta = rw_beta
        self.rw_threshold = rw_threshold
        self.rw_minsize = rw_minsize
        self.rw_sigma = rw_sigma

        # Post processing size threshold
        self.post_minsize = post_minsize

    def __call__(self, predictions_path):

        # Generate some random affinities:
        pmaps = h5py.File(predictions_path, "r")
        pmaps = np.array(pmaps["predictions"][0], dtype=np.float32)

        runtime = time.time()
        segmentation = self.rw(pmaps)

        # run size threshold for rw
        if self.rw_minsize > 0:
            segmentation, _ = apply_size_filter(segmentation, pmaps, self.rw_minsize)

        """
        segmentation = segment_volume(pmaps, segmentation)

        # run size threshold for multicut
        if self.post_minsize > self.rw_minsize:
            segmentation, _ = apply_size_filter(segmentation, pmaps, self.post_minsize)
        """

        runtime = time.time() - runtime
        os.makedirs(os.path.dirname(predictions_path) + "/" + self.save_directory + "/", exist_ok=True)
        h5_file_path = (os.path.dirname(predictions_path) +
                        "/" + self.save_directory + "/" + os.path.basename(predictions_path))

        h5_file_path = os.path.splitext(h5_file_path)[0] + "_randomwalker_os" + ".h5"
        print(h5_file_path)

        self.runtime = runtime
        self._log_params(h5_file_path)

        # Save output results
        with h5py.File(h5_file_path, "w") as file:
            file.create_dataset("segmentation", data=segmentation.astype(np.uint16), compression='gzip')

        print("RW took {} s".format(runtime))
        return h5_file_path

    def _log_params(self, file):
        import yaml
        file = os.path.splitext(file)[0] + ".yaml"
        dict_file = {"algorithm": self.__class__.__name__}

        for name, value in self.__dict__.items():
            dict_file[name] = value

        with open(file, "w") as f:
            f.write(yaml.dump(dict_file))

    def dt_randomwalker(self, pmaps):
        # threshold the input and compute distance transform
        thresholded = (pmaps > self.rw_threshold).astype('uint32')
        dt = vigra.filters.distanceTransform(thresholded, pixel_pitch=None)
        # compute seeds from maxima of the (smoothed) distance transform
        if self.rw_sigma > 0:
            dt = ff.gaussianSmoothing(dt, self.rw_sigma)
        compute_maxima = vigra.analysis.localMaxima if dt.ndim == 2 else vigra.analysis.localMaxima3D
        seeds = compute_maxima(dt, marker=np.nan, allowAtBorder=True, allowPlateaus=True)
        seeds = np.isnan(seeds)
        seeds = vigra.analysis.labelMultiArrayWithBackground(seeds.view('uint8'))

        if pmaps.ndim == 3:
            segmentation = random_walker_algorithm_3d(pmaps,
                                                      beta=self.rw_beta,
                                                      seeds_mask=seeds,
                                                      solving_mode="multi_grid")
        elif pmaps.ndim == 2:
            # segmentation = random_walker(pmaps, seeds, beta=self.rw_beta)
            segmentation = random_walker_algorithm_2d(pmaps,
                                                      beta=self.rw_beta,
                                                      seeds_mask=seeds,
                                                      solving_mode="direct")
        else:
            raise NotImplementedError
        return segmentation

    def dt_rw_2d(self, pmaps):
        # Axis 0 is z assumed!!!
        rw = np.zeros_like(pmaps).astype(np.uint32)
        max_idx = 1
        for i in range(pmaps.shape[0]):
            timer2 = time.time()
            _pmaps = pmaps[i]
            shape = _pmaps.shape

            _pmaps = zoom(_pmaps, zoom=(512/shape[0], 512/shape[1]))

            _rw = self.dt_randomwalker(_pmaps)
            _rw = _rw + max_idx

            _rw = zoom(_rw, zoom=(shape[0]/_rw.shape[0], shape[1]/_rw.shape[1]), order=0)

            max_idx = _rw.max()
            rw[i] = _rw
            print(i, _pmaps.shape, time.time() -timer2)
        return rw

    def dt_rw_2d_blocks(self, pmaps):
        # Axis 0 is z assumed!!!
        rw = np.zeros_like(pmaps).astype(np.uint32)
        if pmaps.shape[1] == 513 or pmaps.shape[2] == 513:
            max_idx, b_split = 1, 513
        else:
            max_idx, b_split = 1, 512

        for i in range(pmaps.shape[0]):
            _pmaps = pmaps[i]
            if i % 10 == 0:
                print("slice %d of %d" % (i, pmaps.shape[0]))

            blocks = [np.hsplit(vsplit, np.arange(b_split, _pmaps.shape[1], b_split))
                      for vsplit in np.vsplit(_pmaps, np.arange(b_split, _pmaps.shape[0], b_split))]

            # Block wise rw
            rw_tmp0 = []
            for bb in blocks:
                rw_tmp1 = []
                for b in bb:
                    _rw = self.dt_randomwalker(b)
                    _rw = _rw + max_idx
                    max_idx = _rw.max()
                    rw_tmp1.append(_rw)
                rw_tmp0.append(rw_tmp1)

            rw[i] = np.block(rw_tmp0)
        return rw
