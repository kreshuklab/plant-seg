import os

import yaml

from plantseg import plantseg_global_path
from ..models.checkmodels import check_models


def create_predict_config(paths, _config):
    """ Creates the configuration file needed for running the neural network inference"""

    # Load template config
    import torch
    config = yaml.load(open(os.path.join(plantseg_global_path, "resources", "config_predict_template.yaml"), 'r'),
                       Loader=yaml.FullLoader)

    # Add patch and stride size
    if "patch" in _config.keys():
        config["loaders"]["test"]["slice_builder"]["patch_shape"] = _config["patch"]
    if "stride" in _config.keys():
        config["loaders"]["test"]["slice_builder"]["stride_shape"] = _config["stride"]

    # Add paths to raw data
    config["loaders"]["test"]["file_paths"] = paths

    # Add correct device for inference
    if _config["device"] == 'cuda':
        config["device"] = torch.device("cuda:0")
    elif _config["device"] == 'cpu':
        config["device"] = torch.device("cpu")
    else:
        raise NotImplementedError

    # check if all files are in the data directory (~/.plantseg_models/)
    check_models(_config['model_name'], update_files=_config['model_update'])

    # Add model path
    home = os.path.expanduser("~")
    config["model_path"] = os.path.join(home,
                                        ".plantseg_models",
                                        _config['model_name'],
                                        f"{_config['version']}_checkpoint.pytorch")

    # Load train config and add missing info
    config_train = yaml.load(open(os.path.join(home,
                                        ".plantseg_models",
                                        _config['model_name'],
                                        "config_train.yml"), 'r'),
                             Loader=yaml.FullLoader)
    #
    for key, value in config_train["model"].items():
        config["model"][key] = value

    config["model_name"] = _config["model_name"]
    return config
