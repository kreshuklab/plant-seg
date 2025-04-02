import logging

from plantseg import PATH_PREDICT_TEMPLATE
from plantseg.core.zoo import model_zoo
from plantseg.functionals.prediction.utils.array_dataset import ArrayDataset
from plantseg.functionals.prediction.utils.slice_builder import SliceBuilder
from plantseg.training.augs import get_test_augmentations
from plantseg.utils import load_config

logger = logging.getLogger(__name__)


def get_predict_template():
    predict_template = load_config(PATH_PREDICT_TEMPLATE)
    return predict_template


def get_array_dataset(
    raw,
    model_name,
    patch,
    stride_ratio,
    halo_shape,
    multichannel,
    global_normalization=True,
):
    if model_name == "UNet2D":
        if patch[0] != 1:
            logger.warning(
                f"Incorrect z-dimension in the patch_shape for the 2D UNet prediction. {patch[0]}"
                f" was given, but has to be 1. Setting to 1"
            )
            patch = (1, patch[1], patch[2])

    if global_normalization:
        augs = get_test_augmentations(raw)
    else:
        # normalize with per patch statistics
        augs = get_test_augmentations(None)

    stride = get_stride_shape(patch, stride_ratio)
    slice_builder = SliceBuilder(
        raw, label_dataset=None, patch_shape=patch, stride_shape=stride
    )
    return ArrayDataset(
        raw,
        slice_builder,
        augs,
        halo_shape=halo_shape,
        multichannel=multichannel,
        verbose_logging=False,
    )


def get_patch_halo(model_name):
    predict_template = get_predict_template()
    patch_halo = predict_template["predictor"]["patch_halo"]

    config_train = model_zoo.get_model_config_by_name(model_name)
    if config_train["model"]["name"] == "UNet2D":
        patch_halo[0] = 0

    return patch_halo


def get_stride_shape(patch_shape, stride_ratio=0.75):
    # striding MUST be >=1
    return [max(int(p * stride_ratio), 1) for p in patch_shape]
