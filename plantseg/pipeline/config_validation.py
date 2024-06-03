import importlib
import os

import numpy as np
import torch
import yaml

from plantseg.pipeline import gui_logger
from plantseg import PATH_RAW2SEG_TEMPLATE
from plantseg.predictions.functional.utils import get_stride_shape
from plantseg.segmentation.utils import SUPPORTED_ALGORITHMS
from plantseg.models.zoo import model_zoo


deprecated_keys = {'param': 'filter_param'}
special_keys = {
    'key',
    'key_nuclei',
    'channel',
    'channel_nuclei',
    'nuclei_predictions_path',
    'is_segmentation',
}


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
    # some of the legacy_gui int come as strings
    value = int(value) if isinstance(value, str) else value
    return _is_type(key, value, fallback=fallback, check_types=int, text="integer")


def is_binary(key, value, fallback=None):
    return _is_type(key, value, fallback=fallback, check_types=bool, text="bool")


def is_list(key, value, fallback=None):
    return _is_type(key, value, fallback=fallback, check_types=(list, tuple), text="list (or tuple)")


def is_length3(key, value, fallback=None):
    if len(value) != 3:
        _error_message("value must be a list of length 3", key, value, fallback)
        return fallback
    else:
        return value


def iterative_is_float(key, value, fallback=None):
    return [is_float(key, v, f) for v, f in zip(value, fallback)]


def iterative_is_int(key, value, fallback=None):
    return [is_int(key, v, f) for v, f in zip(value, fallback)]


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
    _list_models = model_zoo.list_models()
    if value not in _list_models and not model_zoo.check_models(value):
        _error_message(f"value must be one of {_list_models}", key, value, fallback)
        return fallback
    else:
        return value


def check_cuda(key, value, fallback):
    if value == 'cuda' and (not torch.cuda.is_available()):
        _error_message("torch can not detect a valid cuda device", key, value, fallback)
        return fallback
    return value


def is_segmentation(key, value, fallback):
    if value not in SUPPORTED_ALGORITHMS:
        _error_message(f"value must be one of {SUPPORTED_ALGORITHMS}", key, value, fallback)
        return fallback
    else:
        return value


def is_0to1(key, value, fallback):
    if value >= 1.0 or value <= 0:
        _error_message("value must be between 0 and 1", key, value, fallback)
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
        if out is None:
            gui_logger.warning(f"key: '{key}' is missing, plant-seg is trying to use a default.")

        for check in self.check_list:
            out = check(key, out, self.fallback)

        if out is None:
            raise RuntimeError(f"key: '{key}' is required, plant-seg can not run use default for '{key}'.")

        return out


def load_template():
    def _check(loader, node):
        node = loader.construct_mapping(node, deep=True)
        if type(node) is dict:
            return Check(node)
        else:
            raise NotImplementedError("!check constructor must be dict or list.")

    yaml.add_constructor('!check', _check)
    with open(PATH_RAW2SEG_TEMPLATE, 'r') as f:
        return yaml.full_load(f)


def recursive_config_check(config, template):
    # check if deprecated keys are used
    for d_key in deprecated_keys.keys():
        correct_key = deprecated_keys[d_key]

        if d_key in config.keys() and correct_key not in config.keys():
            gui_logger.warning(f"Deprecated config warning. You are using an old version of the config file. "
                               f"key: '{d_key}' has been renamed '{correct_key}'")
            config[correct_key] = config[d_key]
            del config[d_key]

    for key, value in template.items():

        # check if key exist
        if key not in config:
            config[key] = None

        # perform checks from template
        if isinstance(value, Check):
            config[key] = value(key, config[key])

        # recursively go trough all inner-dictionaries
        elif isinstance(value, dict):
            config[key] = recursive_config_check(config[key], template[key])

    return config


def check_scaling_factor(config):
    """
    This function check if all scaling factors are correctly setup
    """
    pre_rescaling = config["preprocessing"]["factor"]
    post_pred_rescaling = config["cnn_postprocessing"]["factor"]
    post_seg_rescaling = config["segmentation_postprocessing"]["factor"]
    pre_inverse_rescaling = [1.0 / f for f in pre_rescaling]
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


def check_patch_and_stride(config):
    """
    This function check if patch and stride are large enough, if not the case will raise a warning and the predictions
    will have empty slices.
    """
    _stride = config["cnn_prediction"].get("stride_ratio", 0.75)
    patch = config["cnn_prediction"]["patch"]
    axis = ['z', 'x', 'y']
    stride = get_stride_shape(patch, _stride) if isinstance(_stride, float) else _stride
    for _ax, _patch, _stride in zip(axis, patch, stride):
        test_z = _ax == 'z' and 1 < _patch - _stride <= 8
        test_x = _ax == 'x' and _patch - _stride <= 16
        test_y = _ax == 'y' and _patch - _stride <= 16
        if test_z or test_x or test_y:
            gui_logger.warning(f"Stride along {_ax} axis (axis order zxy) is too large, "
                               f"this might lead to empty strides artifacts in the cnn predictions. "
                               f"Please try to either reduce the stride or to increase the patch size.")
    return config


def reverse_recursive_config_check(template, config):
    # check if deprecated keys are used
    for key, value in config.items():
        # check if key exist
        if (key not in template) and (key not in special_keys):
            raise RuntimeError(f"Unknown key: '{key}', please remove it from the config file to run plantseg.")

        if isinstance(value, dict):
            reverse_recursive_config_check(template[key], config[key])

    return None


def config_validation(config):
    # check keys from template
    template = load_template()
    config = recursive_config_check(config, template)

    # additional tests:
    config = check_scaling_factor(config)
    config = check_patch_and_stride(config)

    # reverse check
    reverse_recursive_config_check(template, config)
    return config
