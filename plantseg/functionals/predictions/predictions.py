import logging
from pathlib import Path

import numpy as np
import torch

from plantseg.core.zoo import model_zoo
from plantseg.functionals.dataprocessing.dataprocessing import ImageLayout, fix_layout_to_CZYX, fix_layout_to_ZYX
from plantseg.functionals.predictions.utils.array_dataset import ArrayDataset
from plantseg.functionals.predictions.utils.array_predictor import ArrayPredictor
from plantseg.functionals.predictions.utils.size_finder import find_patch_and_halo_shapes, find_patch_shape
from plantseg.functionals.predictions.utils.slice_builder import SliceBuilder
from plantseg.functionals.predictions.utils.utils import get_stride_shape
from plantseg.training.augs import get_test_augmentations

logger = logging.getLogger(__name__)


def unet_predictions(
    raw: np.ndarray,
    input_layout: ImageLayout,
    model_name: str | None,
    model_id: str | None,
    patch: tuple[int, int, int] | None = None,
    patch_halo: tuple[int, int, int] | None = None,
    single_batch_mode: bool = True,
    device: str = "cuda",
    model_update: bool = False,
    disable_tqdm: bool = False,
    handle_multichannel: bool = False,
    config_path: Path | None = None,
    model_weights_path: Path | None = None,
) -> np.ndarray:
    """Generate predictions from raw data using a specified 3D U-Net model.

    This function handles both single and multi-channel outputs from the model,
    returning appropriately shaped arrays based on the output channel configuration.

    Args:
        raw (np.ndarray): Raw input data.
        Input_layout (ImageLayout): The layout of the input data.
        model_name (str | None): The name of the model to use.
        model_id (str | None): The ID of the model from the BioImage.IO model zoo.
        patch (tuple[int, int, int], optional): Patch size for prediction. Defaults to (80, 160, 160).
        patch_halo (tuple[int, int, int] | None, optional): Halo size around patches. Defaults to None.
        single_batch_mode (bool, optional): Whether to use a single batch for prediction. Defaults to True.
        device (str, optional): The computation device ('cpu', 'cuda', etc.). Defaults to 'cuda'.
        model_update (bool, optional): Whether to update the model to the latest version. Defaults to False.
        disable_tqdm (bool, optional): If True, disables the tqdm progress bar. Defaults to False.
        handle_multichannel (bool, optional): If True, handles multi-channel output properly. Defaults to False.
        config_path (Path | None, optional): Path to the model configuration file. Defaults to None.
        model_weights_path (Path | None, optional): Path to the model weights file. Defaults to None.

    Returns:
        np.ndarray: The predicted boundaries as a 3D (Z, Y, X) or 4D (C, Z, Y, X) array, normalized between 0 and 1.

    Raises:
        ValueError: If neither `model_name`, `model_id`, nor `config_path` are provided.
    """

    if config_path is not None:  # Safari mode for custom models outside zoos
        logger.info("Safari prediction: Running model from custom config path.")
        model, model_config, model_path = model_zoo.get_model_by_config_path(config_path, model_weights_path)
    elif model_id is not None:  # BioImage.IO zoo mode
        logger.info("BioImage.IO prediction: Running model from BioImage.IO model zoo.")
        model, model_config, model_path = model_zoo.get_model_by_id(model_id)
    elif model_name is not None:  # PlantSeg zoo mode
        logger.info("Zoo prediction: Running model from PlantSeg official zoo.")
        model, model_config, model_path = model_zoo.get_model_by_name(model_name, model_update=model_update)
    else:
        raise ValueError("Either `model_name` or `model_id` or `model_path` must be provided.")
    state = torch.load(model_path, map_location="cpu")

    if "model_state_dict" in state:  # Model weights format may vary between versions
        state = state["model_state_dict"]
    model.load_state_dict(state)

    if patch_halo is None:
        try:
            logger.info("Computing theoretical minimum halo from model.")
            patch_halo = model_zoo.compute_3D_halo_for_pytorch3dunet(model)
        except Exception:
            logger.warning("Could not compute halo from model. Using 0 halo size, you may experience edge artifacts.")
            patch_halo = (0, 0, 0)

    if patch is None:
        maximum_patch_shape = find_patch_shape(model, model_config["in_channels"], device)
        patch, patch_halo = find_patch_and_halo_shapes(raw.shape, maximum_patch_shape, patch_halo, both_sides=False)

    print(f"For raw in shape {raw.shape}, Patch shape: {patch}", f"Patch halo shape: {patch_halo}")

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
        raw = fix_layout_to_CZYX(raw, input_layout)
        multichannel_input = True
    else:
        raw = fix_layout_to_ZYX(raw, input_layout)
        multichannel_input = False
    raw = raw.astype("float32")
    augs = get_test_augmentations(raw)  # using full raw to compute global normalization mean and std
    stride = get_stride_shape(patch)
    slice_builder = SliceBuilder(raw, label_dataset=None, patch_shape=patch, stride_shape=stride)
    test_dataset = ArrayDataset(
        raw, slice_builder, augs, halo_shape=patch_halo, multichannel=multichannel_input, verbose_logging=False
    )

    pmaps = predictor(test_dataset)  # pmaps either (C, Z, Y, X) or (C, Y, X)
    return pmaps
