from typing import Tuple, Optional
from pathlib import Path

import numpy as np
import torch

from plantseg.loggers import gui_logger
from plantseg.models.zoo import model_zoo
from plantseg.training.augs import get_test_augmentations
from plantseg.functionals.dataprocessing.dataprocessing import fix_input_shape_to_ZYX, fix_input_shape_to_CZYX
from plantseg.functionals.predictions.utils.array_dataset import ArrayDataset
from plantseg.functionals.predictions.utils.array_predictor import ArrayPredictor
from plantseg.functionals.predictions.utils.slice_builder import SliceBuilder
from plantseg.functionals.predictions.utils.utils import get_patch_halo, get_stride_shape


def unet_predictions(
    raw: np.ndarray,
    model_name: Optional[str],
    model_id: Optional[str],
    patch: Tuple[int, int, int] = (80, 160, 160),
    single_batch_mode: bool = True,
    device: str = "cuda",
    model_update: bool = False,
    disable_tqdm: bool = False,
    handle_multichannel: bool = False,
    config_path: Optional[Path] = None,
    model_weights_path: Optional[Path] = None,
    **kwargs,
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
        pmap (np.ndarray): The predicted boundaries as a 3D (Z, Y, X) or 4D (C, Z, Y, X) array, normalized between 0 and 1.
    """
    if config_path is not None:  # Safari mode for custom models outside zoos
        gui_logger.info("Safari prediction: Running model from custom config path.")
        model, model_config, model_path = model_zoo.get_model_by_config_path(config_path, model_weights_path)
    elif model_id is not None:  # BioImage.IO zoo mode
        gui_logger.info("BioImage.IO prediction: Running model from BioImage.IO model zoo.")
        model, model_config, model_path = model_zoo.get_model_by_id(model_id)
    elif model_name is not None:  # PlantSeg zoo mode
        gui_logger.info("Zoo prediction: Running model from PlantSeg official zoo.")
        model, model_config, model_path = model_zoo.get_model_by_name(model_name, model_update=model_update)
    else:
        raise ValueError("Either `model_name` or `model_id` or `model_path` must be provided.")
    state = torch.load(model_path, map_location="cpu")

    if "model_state_dict" in state:  # Model weights format may vary between versions
        state = state["model_state_dict"]
    model.load_state_dict(state)

    patch_halo = kwargs["patch_halo"] if "patch_halo" in kwargs else get_patch_halo(model_name)  # lazy else statement

    predictor = ArrayPredictor(
        model=model,
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        device=device,
        patch=patch,
        patch_halo=patch_halo,
        single_batch_mode=single_batch_mode,
        headless=False,
        verbose_logging=False,
        disable_tqdm=disable_tqdm,
    )

    if int(model_config["in_channels"]) > 1:  # if multi-channel input
        raw = fix_input_shape_to_CZYX(raw)
        multichannel_input = True
    else:
        raw = fix_input_shape_to_ZYX(raw)
        multichannel_input = False
    raw = raw.astype("float32")
    augs = get_test_augmentations(raw)  # using full raw to compute global normalization mean and std
    stride = get_stride_shape(patch)
    slice_builder = SliceBuilder(raw, label_dataset=None, patch_shape=patch, stride_shape=stride)
    test_dataset = ArrayDataset(
        raw, slice_builder, augs, halo_shape=patch_halo, multichannel=multichannel_input, verbose_logging=False
    )

    pmaps = predictor(test_dataset)  # pmaps either (C, Z, Y, X) or (C, Y, X)

    if (
        int(model_config["out_channels"]) > 1 and handle_multichannel
    ):  # if multi-channel output and who called this function can handle it
        gui_logger.warn(f"`unet_predictions()` has `handle_multichannel`={handle_multichannel}")
        pmaps = fix_input_shape_to_CZYX(pmaps)  # make (C, Y, X) to (C, 1, Y, X) and keep (C, Z, Y, X) unchanged
    else:  # otherwise use old mechanism
        pmaps = fix_input_shape_to_ZYX(pmaps[0])

    return pmaps
