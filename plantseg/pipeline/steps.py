import os
from abc import ABC

import numpy as np
import yaml

from plantseg.dataprocessing.functional.dataprocessing import normalize_01, fix_input_shape
from plantseg.io import smart_load, create_tiff, create_h5
from plantseg.pipeline import gui_logger
from plantseg.pipeline.utils import SUPPORTED_TYPES


class GenericPipelineStep:
    """
    Base class for a single step of a pipeline

    Args:
        input_paths (iterable): paths to the files to be processed
        channel (int): channel to be used from the input volume, None if no C dimension
        output_type (str): numpy dtype or the output
        save_directory (str): relative dir where the output files will be saved
        file_suffix (str): suffix added to the output files
        out_ext (str): output file extension
        state (bool): if True the step is enabled
        h5_output_key (str): output H5 dataset, if None the input key will be used
        save_raw (bool): save raw input in the output H5
    """

    def __init__(self, input_paths, input_type, output_type, save_directory, channel=None,
                 file_suffix="", out_ext=".h5", state=True, h5_output_key=None, save_raw=False):
        assert isinstance(input_paths, list)
        assert len(input_paths) > 0, "Input file paths cannot be empty"
        assert input_type in SUPPORTED_TYPES
        assert output_type in SUPPORTED_TYPES
        assert save_directory is not None

        self.input_paths = input_paths
        self.input_channel = channel
        self.h5_output_key = h5_output_key
        self.output_type = output_type
        self.input_type = input_type
        self.file_suffix = file_suffix
        self.out_ext = out_ext
        self.state = state
        self.save_raw = save_raw

        # create save_directory if it doesn't exist
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
        in_shape = input_data.shape
        out_shape = output_data[0].shape if output_data.ndim == 4 else output_data.shape

        scale_factor = np.array(out_shape) / np.array(in_shape)

        voxel_size = np.array(voxel_size) * scale_factor

        output_path = self._create_output_path(input_path)
        gui_logger.info(f'Saving results in {output_path}')
        self.save_output(output_data, output_path, voxel_size)

        if self.save_raw:
            self.save_raw_dataset(input_path, output_path, voxel_size)

        # return output_path
        return output_path

    def load_stack(self, file_path, check_input_type=True):
        """
        Load data from a given file.

        Args:
            file_path (str): path to the file containing the stack
            check_input_type (bool): check input type and normalize data according to processing type

        Returns:
            tuple(nd.array, tuple(float)): (numpy array containing stack's data, stack's data voxel size)
        """
        _, ext = os.path.splitext(file_path)
        data, (voxel_size, _, key, _) = smart_load(file_path, key=None)
        if self.h5_output_key is None:
            self.h5_output_key = key

        # reshape data to 3D always
        if self.input_channel is not None:
            data = data[self.input_channel]
            gui_logger.warning(f"Input channel {self.input_channel} was selected, assuming dimension order CZYX or CYX")
        data = np.nan_to_num(data)
        data = self._fix_input_shape(data)

        if check_input_type:
            # normalize data according to processing type
            data = self._adjust_input_type(data)
        return data, voxel_size

    def save_raw_dataset(self, input_path, output_path, voxel_size):
        """
        Looks for the raw input file given the `input_path` and if found saves the raw input into the `output_path` h5 file
        """
        raw_path = self._raw_path(input_path)
        if os.path.exists(raw_path):
            raw, _ = smart_load(raw_path, key='raw')
            create_h5(output_path, stack=raw, key='raw', voxel_size=voxel_size, mode='a')

        else:
            gui_logger.warning(f'Cannot save raw input: {raw_path} not found')

    @staticmethod
    def _fix_input_shape(data):
        shape_before = data.shape
        data = fix_input_shape(data)
        shape_after = data.shape
        if shape_before != shape_after:
            gui_logger.warning(f"Input shape {shape_before} was fixed to {shape_after}")
        return data

    def _adjust_input_type(self, data):
        if self.input_type == "labels":
            return data.astype(np.uint16)

        elif self.input_type in ["data_float32", "data_uint8"]:
            data = data.astype(np.float32)
            data = self._normalize_01(data)
            return data

    @staticmethod
    def _normalize_01(data):
        return normalize_01(data)

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
            key = self.h5_output_key if self.h5_output_key is not None else 'raw'
            create_h5(output_path, stack=data, key=key, voxel_size=voxel_size, mode='w')

        elif ext == ".tiff":
            create_tiff(output_path, stack=data, voxel_size=voxel_size)

        self._log_params(output_path)

    def _adjust_output_type(self, data):
        if self.output_type == "labels":
            return data.astype(np.uint16)

        elif self.output_type == "data_float32":
            data = self._normalize_01(data)
            return data.astype(np.float32)

        elif self.output_type == "data_uint8":
            data = self._normalize_01(data)
            data = data * np.iinfo(np.uint8).max
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


class AbstractSegmentationStep(GenericPipelineStep, ABC):
    def __init__(self, input_paths, save_directory, file_suffix, state, channel=None):
        super().__init__(
            input_paths=input_paths,
            channel=channel,
            input_type="data_float32",
            output_type="labels",
            save_directory=save_directory,
            file_suffix=file_suffix,
            out_ext=".h5",
            state=state,
            h5_output_key='segmentation',
        )
        self.prediction_channel = None
