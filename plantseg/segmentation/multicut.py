import os
import glob
import numpy as np
import time
import h5py
import nifty
import nifty.graph.rag as nrag
from elf.segmentation.watershed import distance_transform_watershed, apply_size_filter
from elf.segmentation.features import compute_rag
from elf.segmentation.multicut import multicut_kernighan_lin, transform_probabilities_to_costs


class MulticutFromPmaps:
    def __init__(self,
                 predictions_paths,
                 save_directory="MultiCut",
                 multicut_beta=0.5,
                 run_ws=True,
                 ws_2D=True,
                 ws_threshold=0.5,
                 ws_minsize=50,
                 ws_sigma=2.0,
                 ws_w_sigma=0,
                 post_minsize=50,
                 n_threads=6):

        # name subdirectory created for the segmentation file + generic config
        self.predictions_paths = predictions_paths
        self.save_directory = save_directory
        self.n_threads = n_threads

        # Multicut Parameters
        self.multicut_beta = multicut_beta

        # Watershed parameters
        self.run_ws = run_ws
        self.ws_2D = ws_2D
        self.ws_threshold = ws_threshold
        self.ws_minsize = ws_minsize
        self.ws_sigma = ws_sigma
        self.ws_w_sigma = ws_w_sigma

        # Post processing size threshold
        self.post_minsize = post_minsize

    def __call__(self,):

        # Generate some random affinities:
        for predictions_path in self.predictions_paths:
            pmaps = h5py.File(predictions_path, "r")
            pmaps = np.array(pmaps["predictions"][0], dtype=np.float32)

            runtime = time.time()
            segmentation = self.segment_volume(pmaps)

            if self.post_minsize > self.ws_minsize:
                segmentation, _ = apply_size_filter(segmentation, pmaps, self.post_minsize)

            runtime = time.time() - runtime

            os.makedirs(os.path.dirname(predictions_path) + "/" + self.save_directory + "/", exist_ok=True)
            h5_file_path = (os.path.dirname(predictions_path) +
                            "/" + self.save_directory + "/" + os.path.basename(predictions_path))

            h5_file_path = os.path.splitext(h5_file_path)[0] + "_multicut" + ".h5"

            self.runtime = runtime
            self._log_params(h5_file_path)

            # Save output results
            with h5py.File(h5_file_path, "w") as file:
                file.create_dataset("segmentation", data=segmentation.astype(np.uint16), compression='gzip')

            print("Clustering took {} s".format(runtime))

    def _log_params(self, file):
        import yaml
        file = os.path.splitext(file)[0] + ".yaml"
        dict_file = {"algorithm": self.__class__.__name__}

        for name, value in self.__dict__.items():
            dict_file[name] = value

        with open(file, "w") as f:
            f.write(yaml.dump(dict_file))

    def segment_volume(self, pmaps):
        #if self.run_ws:
        if self.ws_2D:
            ws = self.ws_dt_2D(pmaps)
            #rint("return ws")
            #return ws
        else:
            ws, _ = distance_transform_watershed(pmaps, self.ws_threshold, self.ws_sigma, min_size=self.ws_minsize)

        rag = compute_rag(ws, 1)
        # Computing edge features
        features = nrag.accumulateEdgeMeanAndLength(rag, pmaps, numberOfThreads=1)  # DO NOT CHANGE numberOfThreads
        probs = features[:, 0]  # mean edge prob
        edge_sizes = features[:, 1]
        # Prob -> edge costs
        costs = transform_probabilities_to_costs(probs, edge_sizes=edge_sizes, beta=self.multicut_beta)
        # Creating graph
        graph = nifty.graph.undirectedGraph(rag.numberOfNodes)
        graph.insertEdges(rag.uvIds())
        # Solving multicut
        node_labels = multicut_kernighan_lin(graph, costs)
        return nifty.tools.take(node_labels, ws)

    def ws_dt_2D(self, pmaps):
        # Axis 0 is z assumed!!!
        ws = np.zeros_like(pmaps)
        max_idx = 1
        for i in range(pmaps.shape[0]):
            _pmaps = pmaps[i]
            _ws, _ = distance_transform_watershed(_pmaps, self.ws_threshold, self.ws_sigma, sigma_weights=self.ws_w_sigma, min_size=self.ws_minsize)
            _ws = _ws + max_idx
            max_idx = _ws.max()
            ws[i] = _ws
        return ws
