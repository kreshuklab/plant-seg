import yaml
from plantseg.pipeline import raw2seg_config_template
from functools import partial
from plantseg.pipeline import gui_logger
import importlib
from plantseg.gui import list_models
from plantseg.predictions.utils import STRIDE_ACCURATE, STRIDE_BALANCED, STRIDE_DRAFT
from plantseg.segmentation.utils import SUPPORTED_ALGORITMS
import os
import torch
import numpy as np


def _error_message(error, key, value, fallback):
    _error = f"key: {key} has got value: {value}, but {error}"
    if fallback is None:
        raise RuntimeError(_error)
    else:
        gui_logger.warning(f"{_error}. defaulting default value: {fallback}")


def is_string(key, value, fallback=None):
    if not isinstance(value, str):
        _error_message("value must be a string", key, value, fallback)
        return fallback
    else:
        return value


def is_float(key, value, fallback=None):
    if not isinstance(value, (float, int)):
        _error_message("value must be a float", key, value, fallback)
        return fallback
    else:
        return value


def is_int(key, value, fallback=None):
    if isinstance(value, str):
        # some of the gui int come as strings
        value = int(value)

    if not isinstance(value, int):
        _error_message("value must be a int", key, value, fallback)
        return fallback
    else:
        return value


def is_binary(key, value, fallback=None):
    if not isinstance(value, bool):
        _error_message("value must be a bool", key, value, fallback)
        return fallback
    else:
        return value


def is_3float_tuple(key, value, fallback=None):
    if not isinstance(value, (list, tuple)) and len(value) == 3:
        _error_message("value must be a list of length 3", key, value, fallback)
        return fallback

    for v in value:
        if not isinstance(v, (float, int)):
            _error_message("value elements should be float or int", key, value, fallback)
            return fallback

    return value


def is_3int_tuple(key, value, fallback=None):
    if not isinstance(value, (list, tuple)) and len(value) == 3:
        _error_message("value must be a list of length 3", key, value, fallback)
        return fallback

    for v in value:
        if not isinstance(v, int):
            _error_message("value elements should be int", key, value, fallback)
            return fallback

    return value


def filter_name(key, value, fallback=None):
    filters = ['gaussian', 'median']
    if value not in ['gaussian', 'median']:
        _error_message(f"value must be one of {filters}", key, value, fallback)
        return fallback
    else:
        return value


def is_file_or_dir(key, value, fallback):
    if not (os.path.isdir(value) or os.path.isfile(value)):
        _error_message("value must be a valid file or directory path", key, value, fallback)
        return fallback
    else:
        return value


def model_exist(key, value, fallback):
    _list_models = list_models()
    if value not in _list_models:
        _error_message(f"value must be one of {_list_models}", key, value, fallback)
        return fallback
    else:
        return value


def is_cuda(key, value, fallback):
    if value == 'cuda' and (not torch.cuda.is_available()):
        _error_message(f"torch can not detect a valid cuda device", key, value, fallback)
        return fallback
    return value


def is_stride(key, value, fallback):
    _options = [STRIDE_DRAFT, STRIDE_BALANCED, STRIDE_ACCURATE]
    if isinstance(value, (tuple, list)):
        return is_3int_tuple(key, value, fallback)
    elif value not in _options:
        _error_message(f"value must be one of {_options} or a length 3 list of integers", key, value, fallback)
        return fallback
    else:
        return value


def is_segmentation(key, value, fallback):
    if value not in SUPPORTED_ALGORITMS:
        _error_message(f"value must be one of {SUPPORTED_ALGORITMS}", key, value, fallback)
        return fallback
    else:
        return value


def is_0to1(key, value, fallback):
    if value >= 1.0 or value <= 0:
        _error_message(f"value must be between 0 and 1", key, value, fallback)
        return fallback
    else:
        return value


class Check(object):
    def __init__(self, node):
        check_list = []
        for test in node['tests']:
            m = importlib.import_module('plantseg.pipeline.config_validation')
            check_list.append(getattr(m, test))

        self.check_list = check_list
        self.fallback = node['fallback']

    def __call__(self, key, value):
        out = value
        for check in self.check_list:
            out = check(key, value, self.fallback)

        return out


def load_template():
    def _check(loader, node):
        node = loader.construct_mapping(node, deep=True)
        if type(node) is dict:
            return Check(node)
        else:
            raise NotImplementedError("!check constructor must be dict or list.")

    yaml.add_constructor('!check', _check)

    with open(raw2seg_config_template, 'r') as f:
        return yaml.full_load(f)


def recursive_config_check(config, template):
    for key, value in template.items():
        if key not in config:
            raise RuntimeError(f"key: '{key}' is missing, plant-seg requires '{key}' to run.")

        if isinstance(value, Check):
            config[key] = value(key, config[key])

        elif isinstance(value, dict):
            config[key] = recursive_config_check(config[key], template[key])

    return config


def check_scaling_factor(config):
    pre_rescaling = config["preprocessing"]["factor"]
    post_pred_rescaling = config["cnn_postprocessing"]["factor"]
    post_seg_rescaling = config["segmentation_postprocessing"]["factor"]

    pre_inverse_rescaling = [1.0/f for f in pre_rescaling]
    if not np.allclose(pre_inverse_rescaling, post_pred_rescaling):
        gui_logger.warning(f"Prediction post processing scaling is not set up correctly. "
                           f"To avoid shape mismatch between input and output the "
                           f"'factor' value is corrected to {pre_inverse_rescaling}")

        config["cnn_postprocessing"]["factor"] = pre_inverse_rescaling

    if not np.allclose(pre_inverse_rescaling, post_seg_rescaling):
        gui_logger.warning(f"Segmentation post processing scaling is not set up correctly. "
                           f"To avoid shape mismatch between input and output the "
                           f"'factor' value is corrected to {pre_inverse_rescaling}")

        config["segmentation_postprocessing"]["factor"] = pre_inverse_rescaling

    return config


def config_validation(config):
    template = load_template()
    config = recursive_config_check(config, template)

    # additional tests:
    config = check_scaling_factor(config)

    return config
