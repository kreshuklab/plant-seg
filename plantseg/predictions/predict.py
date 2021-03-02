import importlib
import os
import time

import h5py
import torch
from pytorch3dunet.datasets.utils import get_test_loaders
from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.model import get_model
from plantseg.pipeline import gui_logger
from plantseg.predictions.utils import create_predict_config


def _get_output_file(dataset, model_name, suffix='_predictions'):
    basepath, basename = os.path.split(dataset.file_path)
    basename = f"{os.path.splitext(basename)[0]}{suffix}.h5"
    os.makedirs(os.path.join(basepath, model_name), exist_ok=True)
    return os.path.join(basepath, model_name, basename)


def _get_predictor(model, loader, output_file, config):
    predictor_config = config.get('predictor', {})
    class_name = predictor_config.get('name', 'StandardPredictor')

    m = importlib.import_module('pytorch3dunet.unet3d.predictor')
    predictor_class = getattr(m, class_name)

    return predictor_class(model, loader, output_file, config, **predictor_config)


def _check_patch_size(paths, config):
    axis = ['z', 'x', 'y']
    patch_size = config["patch"]
    valid_paths = []

    for path in paths:
        incorrect_axis = []
        with h5py.File(path, 'r') as f:
            raw_size = f["raw"].shape

        for _ax, _patch_size, _raw_size in zip(axis, patch_size, raw_size):
            if _patch_size > _raw_size:
                incorrect_axis.append(_ax)

        if len(incorrect_axis) > 0:
            gui_logger.warning(f"Incorrect Patch size for {path}.\n Patch size {patch_size} along {incorrect_axis}"
                               f" axis (axis order zxy) is too big for an image of size {raw_size},"
                               f" patch size should be smaller or equal than the raw stack size. \n"
                               f"{path} will be skipped.")
        else:
            valid_paths.append(path)

    if len(valid_paths) == 0:
        raise RuntimeError(f"No valid path found for the patch size specified in the PlantSeg config. \n"
                           f" Patch size should be smaller or equal than the raw stack size.")
    return valid_paths


class UnetPredictions:
    def __init__(self, paths, cnn_config):
        assert isinstance(paths, list)
        # check if all file in paths are large enough for the patch size in the config
        valid_paths = _check_patch_size(paths, cnn_config)
        self.paths = valid_paths
        self.cnn_config = cnn_config
        self.state = cnn_config.get("state", True)

    def __call__(self):
        logger = utils.get_logger('UNet3DPredictor')

        if not self.state:
            # skip network predictions and return input_paths
            gui_logger.info(f"Skipping '{self.__class__.__name__}'. Disabled by the user.")
            return self.paths
        else:
            # create config/download models only when cnn_prediction enabled
            config = create_predict_config(self.paths, self.cnn_config)

            # Create the model
            model = get_model(config)

            # Load model state
            model_path = config['model_path']
            model_name = config["model_name"]

            logger.info(f"Loading model '{model_name}' from {model_path}")
            utils.load_checkpoint(model_path, model)
            logger.info(f"Sending the model to '{config['device']}'")
            model = model.to(config['device'])

            logger.info('Loading HDF5 datasets...')

            # Run prediction
            output_paths = []
            for test_loader in get_test_loaders(config):
                gui_logger.info(f"Running network prediction on {test_loader.dataset.file_path}...")
                runtime = time.time()

                logger.info(f"Processing '{test_loader.dataset.file_path}'...")

                output_file = _get_output_file(test_loader.dataset, model_name)

                predictor = _get_predictor(model, test_loader, output_file, config)

                # run the model prediction on the entire dataset and save to the 'output_file' H5
                predictor.predict()

                # save resulting output path
                output_paths.append(output_file)

                runtime = time.time() - runtime
                gui_logger.info(f"Network prediction took {runtime:.2f} s")

            self._update_voxel_size(self.paths, output_paths)

            # free GPU memory after the inference is finished
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return output_paths

    @staticmethod
    def _update_voxel_size(input_paths, output_paths):
        for in_path, out_path in zip(input_paths, output_paths):
            voxel_size = (1., 1., 1.)
            with h5py.File(in_path, 'r') as f:
                if 'element_size_um' in f['raw'].attrs:
                    voxel_size = f['raw'].attrs['element_size_um']

            with h5py.File(out_path, 'r+') as f:
                f['predictions'].attrs['element_size_um'] = voxel_size
