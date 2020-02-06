import os

import yaml

from plantseg import plantseg_global_path
from ..models.checkmodels import check_models

stride_menu = [("Accurate (slower)", 0.5), ("Balanced", 0.75), ("Draft (faster)", 0.9)]


def create_predict_config(paths, _config):
    """ Creates the configuration file needed for running the neural network inference"""

    # Load template config
    import torch
    config = yaml.load(open(os.path.join(plantseg_global_path, "resources", "config_predict_template.yaml"), 'r'),
                       Loader=yaml.FullLoader)

    # Add patch and stride size
    patch, stride = _config["patch"], _config["stride"]
    if "patch" in _config.keys():
        config["loaders"]["test"]["slice_builder"]["patch_shape"] = patch

    if "stride" in _config.keys():
        stride, _stride = _config["stride"], []
        if type(stride) is list:
            config["loaders"]["test"]["slice_builder"]["stride_shape"] = stride
        elif type(stride) is str:
            for stride_key, stride_factor in stride_menu:
                if stride == stride_key:
                    _stride = [int(p * stride_factor) for p in patch]
            config["loaders"]["test"]["slice_builder"]["stride_shape"] = _stride
        else:
            raise RuntimeError(f"Unsupported stride type: {type(stride)}")

    # Add paths to raw data
    config["loaders"]["test"]["file_paths"] = paths

    # Add correct device for inference
    if _config["device"] == 'cuda':
        config["device"] = torch.device("cuda:0")
    elif _config["device"] == 'cpu':
        config["device"] = torch.device("cpu")
    else:
        raise RuntimeError(f"Unsupported device type: {_config['device']}")

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
