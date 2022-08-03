import os

import requests
import torch
import yaml
from pytorch3dunet.unet3d.model import get_model

from plantseg import plantseg_global_path, PLANTSEG_MODELS_DIR, home_path
from plantseg.pipeline import gui_logger
from plantseg.predictions.array_dataset import ArrayDataset
from plantseg.predictions.array_predictor import ArrayPredictor

# define constant values
CONFIG_TRAIN_YAML = "config_train.yml"
BEST_MODEL_PYTORCH = "best_checkpoint.pytorch"
LAST_MODEL_PYTORCH = "last_checkpoint.pytorch"

STRIDE_ACCURATE = "Accurate (slowest)"
STRIDE_BALANCED = "Balanced"
STRIDE_DRAFT = "Draft (fastest)"

STRIDE_MENU = {
    STRIDE_ACCURATE: 0.5,
    STRIDE_BALANCED: 0.75,
    STRIDE_DRAFT: 0.9
}


def get_train_config(model_name):
    # Load train config and add missing info
    train_config_path = os.path.join(home_path,
                                     PLANTSEG_MODELS_DIR,
                                     model_name,
                                     CONFIG_TRAIN_YAML)
    with open(train_config_path, 'r') as f:
        config_train = yaml.full_load(f)
    return config_train


def get_predict_template():
    predict_template_path = os.path.join(plantseg_global_path,
                                         "resources",
                                         "config_predict_template.yaml")
    with open(predict_template_path, 'r') as f:
        predict_template = yaml.full_load(f)
    return predict_template


def get_model_config(model_name, model_update=False, version='best'):
    check_models(model_name, update_files=model_update)
    config_train = get_train_config(model_name)
    model_config = config_train.pop('model')
    model = get_model(model_config)

    model_path = os.path.join(home_path,
                              PLANTSEG_MODELS_DIR,
                              model_name,
                              f"{version}_checkpoint.pytorch")
    return model, model_config, model_path


def set_device(device):
    # Add correct device for inference
    if device == 'cuda':
        device = torch.device("cuda:0")
    elif device == 'cpu':
        device = torch.device("cpu")
    else:
        raise RuntimeError(f"Unsupported device type: {device}")
    return device


def get_loader_config(model_name, patch, stride, mirror_padding, num_workers=8, global_normalization=True):
    predict_template = get_predict_template()
    loader_config = predict_template.pop('loaders')

    loader_config["num_workers"] = num_workers
    loader_config["mirror_padding"] = mirror_padding

    # Add patch and stride to the config
    loader_config["test"]["slice_builder"]["patch_shape"] = patch
    stride_key, stride_shape = stride, get_stride_shape(patch, "Balanced")

    if type(stride_key) is list:
        loader_config["test"]["slice_builder"]["stride_shape"] = stride_key
    elif type(stride_key) is str:
        stride_shape = get_stride_shape(patch, stride_key)
        loader_config["test"]["slice_builder"]["stride_shape"] = stride_shape
    else:
        raise RuntimeError(f"Unsupported stride type: {type(stride_key)}")

    # Set paths to None
    loader_config["test"]["file_paths"] = None

    config_train = get_train_config(model_name)
    if config_train["model"]["name"] == "UNet2D":
        # make sure that z-pad is 0 for 2d UNet
        mirror_padding[0] = 0
        # make sure to skip the patch size validation for 2d unet
        loader_config["test"]["slice_builder"]["skip_shape_check"] = True

        # z-dim of patch and stride has to be one
        patch_shape = loader_config["test"]["slice_builder"]["patch_shape"]
        stride_shape = loader_config["test"]["slice_builder"]["stride_shape"]

        if patch_shape[0] != 1:
            gui_logger.warning(f"Incorrect z-dimension in the patch_shape for the 2D UNet prediction. {patch_shape[0]}"
                               f" was given, but has to be 1. Defaulting default value: 1")
            loader_config["test"]["slice_builder"]["patch_shape"][0] = 1

        if stride_shape[0] != 1:
            gui_logger.warning(f"Incorrect z-dimension in the stride_shape for the 2D UNet prediction. "
                               f"{stride_shape[0]} was given, but has to be 1. Defaulting default value: 1")
            loader_config["test"]["slice_builder"]["stride_shape"][0] = 1

    dataset_config = {'slice_builder_config': loader_config['test']['slice_builder'],
                      'transformer_config': loader_config['test']['transformer'],
                      'mirror_padding': loader_config['mirror_padding'],
                      'global_normalization': global_normalization
                      }

    return ArrayDataset, dataset_config


def get_predictor_config(model_name):
    predict_template = get_predict_template()
    patch_halo = predict_template['predictor']['patch_halo']

    config_train = get_train_config(model_name)
    if config_train["model"]["name"] == "UNet2D":
        patch_halo[0] = 0

    return ArrayPredictor, {'patch_halo': patch_halo}


def old_create_predict_config(paths, cnn_config):
    """ Creates the configuration file needed for running the neural network inference"""
    # TODO safe remove it
    # Load template config
    prediction_config = yaml.load(
        open(os.path.join(plantseg_global_path, "resources", "config_predict_template.yaml"), 'r'),
        Loader=yaml.FullLoader)

    # update loaders
    prediction_config["loaders"]["num_workers"] = cnn_config.get("num_workers", 8)

    # Add patch and stride to the config
    patch_shape = cnn_config["patch"]
    prediction_config["loaders"]["test"]["slice_builder"]["patch_shape"] = patch_shape
    stride_key, stride_shape = cnn_config["stride"], get_stride_shape(patch_shape, "Balanced")

    if type(stride_key) is list:
        prediction_config["loaders"]["test"]["slice_builder"]["stride_shape"] = stride_key
    elif type(stride_key) is str:
        stride_shape = get_stride_shape(patch_shape, stride_key)
        prediction_config["loaders"]["test"]["slice_builder"]["stride_shape"] = stride_shape
    else:
        raise RuntimeError(f"Unsupported stride type: {type(stride_key)}")

    # Add paths to raw data
    prediction_config["loaders"]["test"]["file_paths"] = paths

    # Add correct device for inference
    if cnn_config["device"] == 'cuda':
        prediction_config["device"] = torch.device("cuda:0")
    elif cnn_config["device"] == 'cpu':
        prediction_config["device"] = torch.device("cpu")
    else:
        raise RuntimeError(f"Unsupported device type: {cnn_config['device']}")

    # check if all files are in the data directory (~/.plantseg_models/) and download if needed
    check_models(cnn_config['model_name'], update_files=cnn_config['model_update'])

    # Add model path
    home = os.path.expanduser("~")
    prediction_config["model_path"] = os.path.join(home,
                                                   PLANTSEG_MODELS_DIR,
                                                   cnn_config['model_name'],
                                                   f"{cnn_config['version']}_checkpoint.pytorch")

    # Load train config and add missing info
    config_train = yaml.full_load(
        open(os.path.join(home,
                          PLANTSEG_MODELS_DIR,
                          cnn_config['model_name'],
                          CONFIG_TRAIN_YAML),
             'r'))

    # Load model configuration
    for key, value in config_train["model"].items():
        prediction_config["model"][key] = value

    # configure halo to be removed from the patches
    patch_halo = cnn_config.get("patch_halo", [8, 16, 16])

    # configure mirror padding
    mirror_padding = cnn_config.get("mirror_padding", [16, 32, 32])

    # adapt for UNet2D
    if prediction_config["model"]["name"] == "UNet2D":
        # make sure that z-pad is 0 for 2d UNet
        mirror_padding[0] = 0
        # make sure to skip the patch size validation for 2d unet
        prediction_config["loaders"]["test"]["slice_builder"]["skip_shape_check"] = True
        # set the right patch_halo
        patch_halo[0] = 0

        # z-dim of patch and stride has to be one
        patch_shape = prediction_config["loaders"]["test"]["slice_builder"]["patch_shape"]
        stride_shape = prediction_config["loaders"]["test"]["slice_builder"]["stride_shape"]

        if patch_shape[0] != 1:
            gui_logger.warning(f"Incorrect z-dimension in the patch_shape for the 2D UNet prediction. {patch_shape[0]}"
                               f" was given, but has to be 1. Defaulting default value: 1")
            patch_shape[0] = 1

        if stride_shape[0] != 1:
            gui_logger.warning(f"Incorrect z-dimension in the stride_shape for the 2D UNet prediction. "
                               f"{stride_shape[0]} was given, but has to be 1. Defaulting default value: 1")
            stride_shape[0] = 1

    prediction_config["predictor"]["patch_halo"] = patch_halo
    prediction_config["loaders"]["mirror_padding"] = mirror_padding

    # Additional attributes
    prediction_config["model_name"] = cnn_config["model_name"]
    prediction_config["model_update"] = cnn_config['model_update']
    prediction_config["state"] = cnn_config["state"]

    return prediction_config


def get_stride_shape(patch_shape, stride_key):
    # striding MUST be >=1
    return [max(int(p * STRIDE_MENU[stride_key]), 1) for p in patch_shape]


def download_model(url, out_dir='.'):
    for file in [CONFIG_TRAIN_YAML, BEST_MODEL_PYTORCH, LAST_MODEL_PYTORCH]:
        with requests.get(f'{url}{file}', allow_redirects=True) as r:
            with open(os.path.join(out_dir, file), 'wb') as f:
                f.write(r.content)


def check_models(model_name, update_files=False):
    """
    Simple script to check and download trained modules
    """
    if os.path.isdir(model_name):
        model_dir = model_name
    else:
        model_dir = os.path.join(os.path.expanduser("~"), PLANTSEG_MODELS_DIR, model_name)
        # Check if model directory exist if not create it
        if ~os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

    model_config_path = os.path.exists(os.path.join(model_dir, CONFIG_TRAIN_YAML))
    model_best_path = os.path.exists(os.path.join(model_dir, BEST_MODEL_PYTORCH))
    model_last_path = os.path.exists(os.path.join(model_dir, LAST_MODEL_PYTORCH))

    # Check if files are there, if not download them
    if (not model_config_path or
            not model_best_path or
            not model_last_path or
            update_files):

        # Read config
        model_file = os.path.join(plantseg_global_path, "resources", "models_zoo.yaml")
        config = yaml.load(open(model_file, 'r'), Loader=yaml.FullLoader)

        if model_name in config:
            url = config[model_name]["path"]

            gui_logger.info(f"Downloading model files from: '{url}' ...")
            download_model(url, out_dir=model_dir)
        else:
            raise RuntimeError(f"Custom model {model_name} corrupted. Required files not found.")
    return True
