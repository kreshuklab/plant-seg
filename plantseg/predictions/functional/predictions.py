from typing import Tuple

import numpy as np
import torch

from plantseg.models.zoo import model_zoo
from plantseg.viewer.logging import napari_formatted_logging
from plantseg.augment.transforms import get_test_augmentations
from plantseg.dataprocessing.functional.dataprocessing import fix_input_shape_to_3D, fix_input_shape_to_4D
from plantseg.predictions.functional.array_dataset import ArrayDataset
from plantseg.predictions.functional.array_predictor import ArrayPredictor
from plantseg.predictions.functional.slice_builder import SliceBuilder
from plantseg.predictions.functional.utils import get_patch_halo, get_stride_shape

def unet_predictions(
    raw: np.ndarray,
    model_name: str,
    patch: Tuple[int, int, int] = (80, 160, 160),
    single_batch_mode: bool = True,
    device: str = 'cuda',
    model_update: bool = False,
    disable_tqdm: bool = False,
    handle_multichannel: bool = False,
    **kwargs
) -> np.ndarray:
    """Generate predictions from raw data using a specified 3D U-Net model.

    This function handles both single and multi-channel outputs from the model,
    returning appropriately shaped arrays based on the output channel configuration.

    Args:
        raw (np.ndarray): Raw input data as a 3D array of shape (Z, Y, X).
        model_name (str): The name of the model to use.
        patch (Tuple[int, int, int], optional): Patch size for prediction. Defaults to (80, 160, 160).
        single_batch_mode (bool, optional): Whether to use a single batch for prediction. Defaults to True.
        device (str, optional): The computation device ('cpu', 'cuda', etc.). Defaults to 'cuda'.
        model_update (bool, optional): Whether to update the model to the latest version. Defaults to False.
        disable_tqdm (bool, optional): If True, disables the tqdm progress bar. Defaults to False.
        handle_multichannel (bool, optional): If True, handles multi-channel output properly. Defaults to False.

    Returns:
        np.ndarray: The predicted boundaries as a 3D (Z, Y, X) or 4D (C, Z, Y, X) array, normalized between 0 and 1.
    """
    model, model_config, model_path = model_zoo.get_model_config(model_name, model_update=model_update)
    state = torch.load(model_path, map_location='cpu')

    if 'model_state_dict' in state:  # Model weights format may vary between versions
        state = state['model_state_dict']
    model.load_state_dict(state)

    patch_halo = kwargs.get('patch_halo', get_patch_halo(model_name))

    predictor = ArrayPredictor(
        model=model,
        in_channels=model_config['in_channels'],
        out_channels=model_config['out_channels'],
        device=device,
        patch=patch,
        patch_halo=patch_halo,
        single_batch_mode=single_batch_mode,
        headless=False,
        verbose_logging=False,
        disable_tqdm=disable_tqdm
    )

    raw = fix_input_shape_to_3D(raw)
    raw = raw.astype('float32')
    augs = get_test_augmentations(raw)  # using full raw to compute global normalization mean and std
    stride = get_stride_shape(patch)
    slice_builder = SliceBuilder(raw, label_dataset=None, patch_shape=patch, stride_shape=stride)
    test_dataset = ArrayDataset(raw, slice_builder, augs, halo_shape=patch_halo, verbose_logging=False)

    pmaps = predictor(test_dataset)  # pmaps either (C, Z, Y, X) or (C, Y, X)

    if int(model_config['out_channels']) > 1 and handle_multichannel:  # if multi-channel output and who called this function can handle it
        napari_formatted_logging(
            f'`unet_predictions()` has `handle_multichannel`={handle_multichannel}',
            thread="unet_predictions",
            level='warning'
        )
        pmaps = fix_input_shape_to_4D(pmaps)  # make (C, Y, X) to (C, 1, Y, X) and keep (C, Z, Y, X) unchanged
    else: # otherwise use old mechanism
        pmaps = fix_input_shape_to_3D(pmaps[0])

    return pmaps
