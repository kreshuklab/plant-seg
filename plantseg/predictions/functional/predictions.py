from typing import Tuple

import numpy as np
import torch

from plantseg.viewer.logging import napari_formatted_logging
from plantseg.augment.transforms import get_test_augmentations
from plantseg.dataprocessing.functional.dataprocessing import fix_input_shape_to_3D, fix_input_shape_to_4D
from plantseg.predictions.functional.array_dataset import ArrayDataset
from plantseg.predictions.functional.array_predictor import ArrayPredictor
from plantseg.predictions.functional.slice_builder import SliceBuilder
from plantseg.predictions.functional.utils import get_model_config, get_patch_halo, \
    get_stride_shape


def unet_predictions(raw: np.array, model_name: str, patch: Tuple[int, int, int] = (80, 160, 160),
                     single_batch_mode: bool = True, device: str = 'cuda', model_update: bool = False,
                     disable_tqdm: bool = False, handle_multichannel = False, **kwargs) -> np.array:
    """
    Predict boundaries predictions from raw data using a 3D U-Net model.
    If the model has single-channel output, then return a 3D array of shape (Z, Y, X).
    If the model has multi-channel output, then return a 4D array of shape (C, Z, Y, X).

    Args:
        raw (np.array): raw data, must be a 3D array of shape (Z, Y, X) normalized between 0 and 1.
        model_name (str): name of the model to use. A complete list of available models can be found here:
        patch (tuple[int, int, int], optional): patch size to use for prediction. Defaults to (80, 160, 160).
        single_batch_mode (bool): if True will use a single batch for prediction. Defaults to True.
        device: (str, optional): device to use for prediction. Must be one of ['cpu', 'cuda', 'cuda:1', etc.].
            Defaults to 'cuda'.
        model_update (bool, optional): if True will update the model to the latest version. Defaults to False.
        disable_tqdm (bool, optional): if True will disable tqdm progress bar. Defaults to False.
        output_ndim (int, optional): output ndim, must be one of [3, 4]. Only use `4` if network output is
            multi-channel 3D pmap. Now `4` only used in `widget_unet_predictions()`.

    Returns:
        np.array: predictions, 4D array of shape (C, Z, Y, X) or 3D array of shape (Z, Y, X) with values between 0 and 1.
            if `out_channels` in model config is greater than 1, then output will be 4D array.
        :param single_batch_mode:
    """
    model, model_config, model_path = get_model_config(model_name, model_update=model_update)
    state = torch.load(model_path, map_location='cpu')
    # ensure compatibility with models trained with pytorch-3dunet
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)

    patch_halo = kwargs.get('patch_halo', None)
    if patch_halo is None:
        patch_halo = get_patch_halo(model_name)
    predictor = ArrayPredictor(model=model, in_channels=model_config['in_channels'],
                               out_channels=model_config['out_channels'], device=device, patch=patch,
                               patch_halo=patch_halo, single_batch_mode=single_batch_mode, headless=False,
                               verbose_logging=False, disable_tqdm=disable_tqdm)

    raw = fix_input_shape_to_3D(raw)
    raw = raw.astype('float32')
    stride = get_stride_shape(patch)
    augs = get_test_augmentations(raw)
    slice_builder = SliceBuilder(raw, label_dataset=None, patch_shape=patch, stride_shape=stride)
    test_dataset = ArrayDataset(raw, slice_builder, augs, halo_shape=patch_halo, verbose_logging=False)

    pmaps = predictor(test_dataset)  # pmaps either (C, Z, Y, X) or (C, Y, X)
    out_channel = int(model_config['out_channels'])

    if out_channel > 1 and handle_multichannel:  # if multi-channel output and who called this function can handle it
        napari_formatted_logging(f'`unet_predictions()` has `handle_multichannel`={handle_multichannel}',
                                 thread="unet_predictions", level='warning')
        pmaps = fix_input_shape_to_4D(pmaps)  # then make (C, Y, X) to (C, 1, Y, X) and keep (C, Z, Y, X) unchanged
    else:  # otherwise use old mechanism
        pmaps = fix_input_shape_to_3D(pmaps[0])
    return pmaps
