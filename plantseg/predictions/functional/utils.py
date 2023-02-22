import os

import torch
from pytorch3dunet.unet3d.model import get_model

from plantseg import plantseg_global_path, PLANTSEG_MODELS_DIR, home_path
from plantseg.utils import load_config
from plantseg.pipeline import gui_logger
from plantseg.predictions.functional.array_dataset import ArrayDataset
from plantseg.predictions.functional.array_predictor import ArrayPredictor
from plantseg.utils import get_train_config, check_models

# define constant values

STRIDE_ACCURATE = "Accurate (slowest)"
STRIDE_BALANCED = "Balanced"
STRIDE_DRAFT = "Draft (fastest)"

STRIDE_MENU = {
    STRIDE_ACCURATE: 0.5,
    STRIDE_BALANCED: 0.75,
    STRIDE_DRAFT: 0.9
}


def get_predict_template():
    predict_template_path = os.path.join(plantseg_global_path,
                                         "resources",
                                         "config_predict_template.yaml")
    predict_template = load_config(predict_template_path)
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


def set_device(device, device_id=0):
    device = device if torch.cuda.is_available() else 'cpu'

    # Add correct device for inference
    if device == 'cuda':
        device = torch.device(f"cuda:{device_id}")
    elif device == 'cpu':
        device = torch.device("cpu")
    else:
        raise RuntimeError(f"Unsupported device type: {device}")
    return device


def get_dataset_config(model_name, patch, stride, mirror_padding, num_workers=8, global_normalization=True):
    predict_template = get_predict_template()
    dataset_config = predict_template.pop('loaders')

    dataset_config["num_workers"] = num_workers
    dataset_config["mirror_padding"] = mirror_padding
    # Add patch and stride to the config
    dataset_config["test"]["slice_builder"]["patch_shape"] = patch
    stride_key, stride_shape = stride, get_stride_shape(patch, "Balanced")

    if type(stride_key) is list:
        dataset_config["test"]["slice_builder"]["stride_shape"] = stride_key
    elif type(stride_key) is str:
        stride_shape = get_stride_shape(patch, stride_key)
        dataset_config["test"]["slice_builder"]["stride_shape"] = stride_shape
    else:
        raise RuntimeError(f"Unsupported stride type: {type(stride_key)}")

    # Set paths to None
    dataset_config["test"]["file_paths"] = None

    config_train = get_train_config(model_name)
    if config_train["model"]["name"] == "UNet2D":
        # make sure that z-pad is 0 for 2d UNet
        dataset_config["mirror_padding"] = [0, mirror_padding[1], mirror_padding[2]]
        # make sure to skip the patch size validation for 2d unet
        dataset_config["test"]["slice_builder"]["skip_shape_check"] = True

        # z-dim of patch and stride has to be one
        patch_shape = dataset_config["test"]["slice_builder"]["patch_shape"]
        stride_shape = dataset_config["test"]["slice_builder"]["stride_shape"]

        if patch_shape[0] != 1:
            gui_logger.warning(f"Incorrect z-dimension in the patch_shape for the 2D UNet prediction. {patch_shape[0]}"
                               f" was given, but has to be 1. Defaulting default value: 1")
            dataset_config["test"]["slice_builder"]["patch_shape"] = (1, patch_shape[1], patch_shape[2])

        if stride_shape[0] != 1:
            gui_logger.warning(f"Incorrect z-dimension in the stride_shape for the 2D UNet prediction. "
                               f"{stride_shape[0]} was given, but has to be 1. Defaulting default value: 1")
            dataset_config["test"]["slice_builder"]["stride_shape"] = (1, stride_shape[1], stride_shape[2])

    dataset_config = {'slice_builder_config': dataset_config['test']['slice_builder'],
                      'transformer_config': dataset_config['test']['transformer'],
                      'mirror_padding': dataset_config['mirror_padding'],
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


def get_stride_shape(patch_shape, stride_key):
    # striding MUST be >=1
    return [max(int(p * STRIDE_MENU[stride_key]), 1) for p in patch_shape]
