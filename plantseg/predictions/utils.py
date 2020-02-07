import os

import torch
import yaml

from plantseg import plantseg_global_path
from plantseg.models.checkmodels import check_models

STRIDE_ACCURATE = "Accurate (slowest)"
STRIDE_BALANCED = "Balanced"
STRIDE_DRAFT = "Draft (fastest)"

STRIDE_MENU = {
    STRIDE_ACCURATE: 0.5,
    STRIDE_BALANCED: 0.75,
    STRIDE_DRAFT: 0.9
}


def create_predict_config(paths, plantseg_config):
    """ Creates the configuration file needed for running the neural network inference"""

    def _stride_shape(patch_shape, stride_key):
        return [int(p * STRIDE_MENU[stride_key]) for p in patch_shape]

    # Load template config
    prediction_config = yaml.load(
        open(os.path.join(plantseg_global_path, "resources", "config_predict_template.yaml"), 'r'),
        Loader=yaml.FullLoader)

    # Add patch and stride to the config
    patch_shape = plantseg_config["patch"]
    prediction_config["loaders"]["test"]["slice_builder"]["patch_shape"] = patch_shape
    stride_key, stride_shape = plantseg_config["stride"], _stride_shape(patch_shape, "Balanced")

    if type(stride_key) is list:
        prediction_config["loaders"]["test"]["slice_builder"]["stride_shape"] = stride_key
    elif type(stride_key) is str:
        stride_shape = _stride_shape(patch_shape, stride_key)
        prediction_config["loaders"]["test"]["slice_builder"]["stride_shape"] = stride_shape
    else:
        raise RuntimeError(f"Unsupported stride type: {type(stride)}")

    # Add paths to raw data
    prediction_config["loaders"]["test"]["file_paths"] = paths

    # Add correct device for inference
    if plantseg_config["device"] == 'cuda':
        prediction_config["device"] = torch.device("cuda:0")
    elif plantseg_config["device"] == 'cpu':
        prediction_config["device"] = torch.device("cpu")
    else:
        raise RuntimeError(f"Unsupported device type: {plantseg_config['device']}")

    # check if all files are in the data directory (~/.plantseg_models/)
    check_models(plantseg_config['model_name'], update_files=plantseg_config['model_update'])

    # Add model path
    home = os.path.expanduser("~")
    prediction_config["model_path"] = os.path.join(home,
                                                   ".plantseg_models",
                                                   plantseg_config['model_name'],
                                                   f"{plantseg_config['version']}_checkpoint.pytorch")

    # Load train config and add missing info
    config_train = yaml.load(open(os.path.join(home,
                                               ".plantseg_models",
                                               plantseg_config['model_name'],
                                               "config_train.yml"), 'r'),
                             Loader=yaml.FullLoader)
    #
    for key, value in config_train["model"].items():
        prediction_config["model"][key] = value

    prediction_config["model_name"] = plantseg_config["model_name"]
    return prediction_config
