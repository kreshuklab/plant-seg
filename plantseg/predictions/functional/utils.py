import os

from plantseg import PATH_PLANTSEG_GLOBAL, DIR_PLANTSEG_MODELS, PATH_HOME
from plantseg.augment.transforms import get_test_augmentations
from plantseg.training.model import get_model
from plantseg.pipeline import gui_logger
from plantseg.predictions.functional.array_dataset import ArrayDataset
from plantseg.predictions.functional.slice_builder import SliceBuilder
from plantseg.utils import get_train_config, check_models
from plantseg.utils import load_config


def get_predict_template():
    predict_template_path = os.path.join(PATH_PLANTSEG_GLOBAL,
                                         "resources",
                                         "config_predict_template.yaml")
    predict_template = load_config(predict_template_path)
    return predict_template


def get_model_config(model_name, model_update=False):
    check_models(model_name, update_files=model_update)
    config_train = get_train_config(model_name)
    model_config = config_train.pop('model')
    model = get_model(model_config)
    model_path = os.path.join(PATH_HOME,
                              DIR_PLANTSEG_MODELS,
                              model_name,
                              "best_checkpoint.pytorch")
    return model, model_config, model_path


def get_array_dataset(raw, model_name, patch, stride_ratio, global_normalization=True):
    if model_name == 'UNet2D':
        if patch[0] != 1:
            gui_logger.warning(f"Incorrect z-dimension in the patch_shape for the 2D UNet prediction. {patch[0]}"
                               f" was given, but has to be 1. Setting to  1")
            patch = (1, patch[1], patch[2])

    if global_normalization:
        augs = get_test_augmentations(raw)
    else:
        # normalize with per patch statistics
        augs = get_test_augmentations(None)

    stride = get_stride_shape(patch, stride_ratio)
    slice_builder = SliceBuilder(raw, label_dataset=None, patch_shape=patch, stride_shape=stride)
    return ArrayDataset(raw, slice_builder, augs, verbose_logging=False)


def get_patch_halo(model_name):
    predict_template = get_predict_template()
    patch_halo = predict_template['predictor']['patch_halo']

    config_train = get_train_config(model_name)
    if config_train["model"]["name"] == "UNet2D":
        patch_halo[0] = 0

    return patch_halo


def get_stride_shape(patch_shape, stride_ratio=0.75):
    # striding MUST be >=1
    return [max(int(p * stride_ratio), 1) for p in patch_shape]
