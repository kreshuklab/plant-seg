import importlib
import os

import numpy as np
import torch
import yaml

from plantseg.gui import list_models
from plantseg.pipeline import gui_logger
from plantseg.pipeline import raw2seg_config_template
from plantseg.predictions.utils import STRIDE_ACCURATE, STRIDE_BALANCED, STRIDE_DRAFT
from plantseg.segmentation.utils import SUPPORTED_ALGORITMS


def _error_message(error, key, value, fallback):
    _error = f"key: {key} has got value: {value}, but {error}"
    if fallback is None:
        raise RuntimeError(_error)
    else:
        gui_logger.warning(f"{_error}. defaulting default value: {fallback}")


def _is_type(key, value, fallback=None, check_types=(str,), text="string"):
    if not isinstance(value, check_types):
        _error_message(f"value must be a {text}", key, value, fallback)
        return fallback
    else:
        return value


def is_string(key, value, fallback=None):
    return _is_type(key, value, fallback=fallback, check_types=str, text="string")


def is_float(key, value, fallback=None):
    return _is_type(key, value, fallback=fallback, check_types=(float, int), text="float (or integer)")


def is_int(key, value, fallback=None):
    # some of the gui int come as strings
    value = int(value) if isinstance(value, str) else value
    return _is_type(key, value, fallback=fallback, check_types=int, text="integer")


def is_binary(key, value, fallback=None):
    return _is_type(key, value, fallback=fallback, check_types=bool, text="bool")


def is_list(key, value, fallback=None):
    return _is_type(key, value, fallback=fallback, check_types=(list, tuple), text="list (or tuple)")


def is_length3(key, value, fallback=None):
    if len(value) == 3:
        _error_message(f"value must be a list of length 3", key, value, fallback)
        return fallback
    else:
        return value


def iterative_is_float(key, value, fallback=None):
    for v in value:
        return is_float(key, v, fallback)


def iterative_is_int(key, value, fallback=None):
    for v in value:
        return is_int(key, v, fallback)


def is_stride(key, value, fallback):
    _options = [STRIDE_DRAFT, STRIDE_BALANCED, STRIDE_ACCURATE]
    if isinstance(value, (tuple, list)):
        value = is_length3(key, value, fallback)
        value = iterative_is_int(key, value, fallback)
        return value

    elif value not in _options:
        _error_message(f"value must be one of {_options} or a length 3 list of integers", key, value, fallback)
        return fallback

    else:
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


def check_cuda(key, value, fallback):
    if value == 'cuda' and (not torch.cuda.is_available()):
        _error_message(f"torch can not detect a valid cuda device", key, value, fallback)
        return fallback
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
        assert 'tests' in node.keys(), 'Test is not configured correctly tests key is missing'
        check_list = []
        for test in node['tests']:
            m = importlib.import_module('plantseg.pipeline.config_validation')
            check_list.append(getattr(m, test))

        self.check_list = check_list
        self.fallback = node.get('fallback', None)

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
