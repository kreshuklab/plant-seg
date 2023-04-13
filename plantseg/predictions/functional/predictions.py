from typing import Tuple

import numpy as np
import torch

from plantseg.augment.transforms import get_test_augmentations
from plantseg.dataprocessing.functional.dataprocessing import fix_input_shape
from plantseg.predictions.functional.array_dataset import ArrayDataset
from plantseg.predictions.functional.array_predictor import ArrayPredictor
from plantseg.predictions.functional.slice_builder import SliceBuilder
from plantseg.predictions.functional.utils import get_model_config, get_patch_halo, \
    get_stride_shape


def unet_predictions(raw: np.array, model_name: str, patch: Tuple[int, int, int] = (80, 160, 160),
                     single_batch_mode: bool = True, device: str = 'cuda', model_update: bool = False,
                     disable_tqdm: bool = False, **kwargs) -> np.array:
    """
    Predict boundaries predictions from raw data using a 3D U-Net model.

    Args:
        raw (np.array): raw data, must be a 3D array of shape (Z, Y, X) normalized between 0 and 1.
        model_name (str): name of the model to use. A complete list of available models can be found here:
        patch (tuple[int, int, int], optional): patch size to use for prediction. Defaults to (80, 160, 160).
        single_batch_mode (bool): if True will use a single batch for prediction. Defaults to True.
        device: (str, optional): device to use for prediction. Must be one of ['cpu', 'cuda', 'cuda:1', etc.].
            Defaults to 'cuda'.
        model_update (bool, optional): if True will update the model to the latest version. Defaults to False.
        disable_tqdm (bool, optional): if True will disable tqdm progress bar. Defaults to False.

    Returns:
        np.array: predictions, 3D array of shape (Z, Y, X) with values between 0 and 1.
        :param single_batch_mode:

    """
    model, model_config, model_path = get_model_config(model_name, model_update=model_update)
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state)

    patch_halo = get_patch_halo(model_name)
    predictor = ArrayPredictor(model=model, in_channels=model_config['in_channels'],
                               out_channels=model_config['out_channels'], device=device, patch=patch,
                               patch_halo=patch_halo, single_batch_mode=single_batch_mode, headless=False,
                               verbose_logging=False, disable_tqdm=disable_tqdm)

    raw = fix_input_shape(raw)
    raw = raw.astype('float32')
    stride = get_stride_shape(patch)
    augs = get_test_augmentations(raw)
    slice_builder = SliceBuilder(raw, label_dataset=None, weight_dataset=None, patch_shape=patch, stride_shape=stride)
    test_dataset = ArrayDataset(raw, slice_builder, augs, verbose_logging=False)

    pmaps = predictor(test_dataset)
    pmaps = fix_input_shape(pmaps[0])
    return pmaps
