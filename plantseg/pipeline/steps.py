import os

import h5py
import numpy as np
import tifffile
import yaml

from plantseg.pipeline import gui_logger
from plantseg.pipeline.utils import read_tiff_voxel_size, read_h5_voxel_size, find_input_key, SUPPORTED_TYPES, \
    TIFF_EXTENSIONS, H5_EXTENSIONS


class GenericPipelineStep:
    """
    Base class for the a single step of a pipeline
    
    Args:
        input_paths (iterable): paths to the files to be processed
        output_type (str): numpy dtype or the output
        save_directory (str): relative dir where the output files will be saved
        file_suffix (str): suffix added to the output files
        out_ext (str): output file extension
        state (bool): if True the step is enabled
        h5_output_key (str): output H5 dataset, if None the input key will be used
        save_raw (bool): save raw input in the output H5
    """

    def __init__(self, input_paths, input_type, output_type, save_directory,
                 file_suffix="", out_ext=".h5", state=True, h5_output_key=None, save_raw=False):
        assert isinstance(input_paths, list)
        assert len(input_paths) > 0, "Input file paths cannot be empty"
        assert input_type in SUPPORTED_TYPES
        assert output_type in SUPPORTED_TYPES
        assert save_directory is not None

        self.input_paths = input_paths
        self.h5_output_key = h5_output_key
        self.output_type = output_type
        self.input_type = input_type
        self.file_suffix = file_suffix
        self.out_ext = out_ext
        self.state = state
        self.save_raw = save_raw

        # create save_directory if doesn't exist
        self.save_directory = os.path.join(os.path.dirname(input_paths[0]), save_directory)
        if self.state:
            os.makedirs(self.save_directory, exist_ok=True)

    def __call__(self):
        if not self.state:
            gui_logger.info(f"Skipping '{self.__class__.__name__}'. Disabled by the user.")
            return self.input_paths
        else:
            return [self.read_process_write(input_path) for input_path in self.input_paths]

    def process(self, input_data):
        """
        Abstract method to be implemented by a concrete PipelineStep

        Args:
            input_data (nd.array): input data to process by the PipelineStep

        Returns:
            output_path (str): path to the file where the results were saved
        """
        raise NotImplementedError

    def read_process_write(self, input_path):
        gui_logger.info(f'Loading stack from {input_path}')
        input_data, voxel_size = self.load_stack(input_path)

        output_data = self.process(input_data)
        # voxel_size may change after pre-/post-processing (i.e. when scaling is used)
        scale_factor = np.array(output_data.shape) / np.array(input_data.shape)
        voxel_size = np.array(voxel_size) * scale_factor

        output_path = self._create_output_path(input_path)
        gui_logger.info(f'Saving results in {output_path}')
        self.save_output(output_data, output_path, voxel_size)

        if self.save_raw:
            self.save_raw_dataset(input_path, output_path, voxel_size)

        # return output_path
        return output_path

    def load_stack(self, file_path):
        """
        Load data from a given file.

        Args:
            file_path (str): path to the file containing the stack

        Returns:
            tuple(nd.array, tuple(float)): (numpy array containing stack's data, stack's data voxel size)
        """
        _, ext = os.path.splitext(file_path)

        if ext in TIFF_EXTENSIONS:
            # load tiff file
            data = tifffile.imread(file_path)
            # parse voxel_size
            voxel_size = read_tiff_voxel_size(file_path)
        elif ext in H5_EXTENSIONS:
            # load data from H5 file
            with h5py.File(file_path, "r") as f:
                h5_input_key = find_input_key(f)
                gui_logger.info(f"Found '{h5_input_key}' dataset inside {file_path}")
                # set h5_output_key to be the same as h5_input_key if h5_output_key not defined
                if self.h5_output_key is None:
                    self.h5_output_key = h5_input_key

                ds = f[h5_input_key]
                data = ds[...]

            # Parse voxel size
            voxel_size = read_h5_voxel_size(file_path)
        else:
            raise RuntimeError(f"Unsupported file extension: {ext}")

        # reshape data to 3D always
        data = np.nan_to_num(data)
        data = self._fix_input_shape(data)

        # normalize data according to processing type
        data = self._adjust_input_type(data)
        return data, voxel_size

    def save_raw_dataset(self, input_path, output_path, voxel_size):
        """
        Looks for the raw input file given the `input_path` and if found saves the raw input into the `output_path` h5 file
        """
        raw_path = self._raw_path(input_path)
        if os.path.exists(raw_path):
            with h5py.File(raw_path, 'r') as f:
                raw = f['raw'][...]
            with h5py.File(output_path, 'r+') as f:
                f.create_dataset('raw', data=raw, compression='gzip')
                # save voxel_size
                f['raw'].attrs['element_size_um'] = voxel_size
        else:
            gui_logger.warning(f'Cannot save raw input: {raw_path} not found')

    @staticmethod
    def _fix_input_shape(data):
        if data.ndim == 2:
            return data.reshape(1, data.shape[0], data.shape[1])

        elif data.ndim == 3:
            return data

        elif data.ndim == 4:
            return data[0]

        else:
            raise RuntimeError(f"Expected input data to be 2d, 3d or 4d, but got {data.ndim}d input")

    def _adjust_input_type(self, data):
        if self.input_type == "labels":
            return data.astype(np.uint16)

        elif self.input_type in ["data_float32", "data_uint8"]:
            data = data.astype(np.float32)
            data = self._normalize_01(data)
            return data

    @staticmethod
    def _normalize_01(data):
        return (data - data.min()) / (data.max() - data.min() + 1e-12)

    def _log_params(self, file):
        file = os.path.splitext(file)[0] + ".yaml"
        dict_file = {"algorithm": self.__class__.__name__}

        for name, value in self.__dict__.items():
            dict_file[name] = value

        with open(file, "w") as f:
            f.write(yaml.dump(dict_file))

    def _create_output_path(self, input_path):
        output_path = os.path.join(self.save_directory, os.path.basename(input_path))
        output_path = os.path.splitext(output_path)[0] + self.file_suffix + self.out_ext
        return output_path

    def save_output(self, data, output_path, voxel_size):
        assert voxel_size is not None and len(voxel_size) == 3
        data = self._adjust_output_type(data)
        _, ext = os.path.splitext(output_path)
        assert ext in [".h5", ".tiff"], f"Unsupported file extension {ext}"

        if ext == ".h5":
            # Save output results as h5
            with h5py.File(output_path, "w") as f:
                if self.h5_output_key is None:
                    # this can happen if input file is tiff and h5_output_key was not specified
                    self.h5_output_key = 'raw'

                f.create_dataset(self.h5_output_key, data=data, compression='gzip')
                # save voxel_size
                f[self.h5_output_key].attrs['element_size_um'] = voxel_size
        elif ext == ".tiff":
            # taken from: https://pypi.org/project/tifffile docs
            z, y, x = data.shape
            data.shape = 1, z, 1, y, x, 1  # dimensions in TZCYXS order
            spacing, y, x = voxel_size
            resolution = (1. / x, 1. / y)
            # Save output results as tiff
            tifffile.imsave(output_path,
                            data=data,
                            dtype=data.dtype,
                            imagej=True,
                            resolution=resolution,
                            metadata={'axes': 'TZCYXS', 'spacing': spacing, 'unit': 'um'})

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

    def _raw_path(self, input_path):
        base, filename = os.path.split(input_path)
        filename = filename.split('_predictions')[0] + '.h5'

        up_levels = 2
        if self.input_type != 'labels':
            up_levels = 1

        for _ in range(up_levels):
            base = os.path.dirname(base)

        return os.path.join(base, filename)


class AbstractSegmentationStep(GenericPipelineStep):
    def __init__(self, input_paths, save_directory, file_suffix, state):
        super().__init__(input_paths=input_paths,
                         input_type="data_float32",
                         output_type="labels",
                         save_directory=save_directory,
                         file_suffix=file_suffix,
                         out_ext=".h5",
                         state=state,
                         h5_output_key='segmentation')
