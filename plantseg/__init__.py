import os
from pathlib import Path

import h5py
import numpy as np
import tifffile
import yaml

# Find the global path of  plantseg
plantseg_global_path = Path(__file__).parent.absolute()

# Create configs directory at startup
home_path = os.path.expanduser("~")
configs_path = os.path.join(home_path, ".plantseg_models", "configs")
os.makedirs(configs_path, exist_ok=True)


# Generic pipeline prototype
class GenericProcessing:
    def __init__(self, predictions_paths, input_type, output_type, save_directory="GenericProcess"):

        # Define paths names, destination directory, output files paths and i/o file type.
        self.predictions_paths = predictions_paths
        self.save_directory = save_directory
        self.outputs_paths = []

        assert input_type in ["labels", "data_float32", "data_uint8"]
        assert output_type in ["labels", "data_float32", "data_uint8"]

        self.input_type = input_type
        self.output_type = output_type

        # file format and datasets keys
        self.tiff_ext = [".tiff", ".tif"]
        self.h5_ext = [".hdf", ".h5", ".hd5"]
        self.h5_key = ["raw", "predictions", "segmentation"]

    def load_stack(self, path):
        # Load file
        _, ext = os.path.splitext(path)
        data = None
        if ext in self.tiff_ext:
            data = tifffile.imread(path)

        elif ext in self.h5_ext:
            with h5py.File(path, "r") as f:
                # Check for h5 dataset
                dataset = None
                for key in self.h5_key:
                    if key in f.keys():
                        dataset = key

                if dataset is None:
                    print("H5 dataset name not understood")
                    raise NotImplementedError

                # Load data on disk
                data = f[dataset][...]

        else:
            print("Data extension not understood")
            raise NotImplementedError

        # reshape data to 3D always
        data = np.nan_to_num(data, nan=0)
        data = self._fix_input_shape(data)

        # normalize data according to processing type
        data = self._adjust_input_type(data)
        return data

    @staticmethod
    def _fix_input_shape(data):
        data_shape, data_dim = data.shape, len(data.shape)
        if data_dim == 2:
            return data.reshape(1, data_shape.shape[0], data_shape.shape[1])

        elif data_dim == 3:
            return data

        elif data_dim == 4:
            return data[0]

        else:
            print("Data dimension not understood, data must be 2d or 3d or 4d")
            raise NotImplementedError

    def _adjust_input_type(self, data):
        if self.input_type == "labels":
            return data.astype(np.uint16)

        elif self.input_type in ["data_float32", "data_uint8"]:
            data = data.astype(np.float32)
            data = self._normalize_01(data)
            return data

        else:
            print("Input type not understood")
            raise NotImplementedError

    def _log_params(self, file):
        file = os.path.splitext(file)[0] + ".yaml"
        dict_file = {"algorithm": self.__class__.__name__}

        for name, value in self.__dict__.items():
            dict_file[name] = value

        with open(file, "w") as f:
            f.write(yaml.dump(dict_file))

    def create_output_path(self, input_path, prefix="", out_ext=".h5"):
        os.makedirs(os.path.join(os.path.dirname(input_path), self.save_directory), exist_ok=True)

        output_path = os.path.join(os.path.dirname(input_path),
                                   self.save_directory,
                                   os.path.basename(input_path))

        output_path = os.path.splitext(output_path)[0] + prefix + out_ext
        return output_path, os.path.isfile(output_path)

    @staticmethod
    def _normalize_01(data):
        return (data - data.min()) / (data.max() - data.min() + 1e-12)

    def save_output(self, data, output_path, dataset="data"):
        data = self._adjust_output_type(data)
        _, ext = os.path.splitext(output_path)

        if ext == ".h5":
            # Save output results as h5
            with h5py.File(output_path, "w") as file:
                file.create_dataset(dataset, data=data, compression='gzip')

        elif ext == ".tiff":
            # Save output results as tiff
            tifffile.imsave(output_path,
                            data=data,
                            dtype=data.dtype,
                            bigtiff=True,
                            resolution=(1, 1),
                            metadata={'spacing': 1, 'unit': 'um'})

        self._log_params(output_path)

    def _adjust_output_type(self, data):
        if self.output_type == "labels":
            return data.astype(np.uint16)

        elif self.output_type == "data_float32":
            data = self._normalize_01(data)
            return data.astype(np.float32)

        elif self.output_type == "data_uint8":
            data = self._normalize_01(data)
            data = (data * np.iinfo(np.uint8).max)
            return data.astype(np.uint8)

        else:
            print("Input type not understood")
            raise NotImplementedError
