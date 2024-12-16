import logging
from pathlib import Path
from typing import assert_never

import numpy as np
import torch
from bioimageio.core.axis import AxisId
from bioimageio.core.prediction import predict
from bioimageio.core.sample import Sample
from bioimageio.core.tensor import Tensor
from bioimageio.spec import load_model_description
from bioimageio.spec.model import v0_4, v0_5
from bioimageio.spec.model.v0_5 import TensorId

from plantseg.core.zoo import model_zoo
from plantseg.functionals.dataprocessing.dataprocessing import ImageLayout, fix_layout_to_CZYX, fix_layout_to_ZYX
from plantseg.functionals.prediction.utils.array_dataset import ArrayDataset
from plantseg.functionals.prediction.utils.array_predictor import ArrayPredictor
from plantseg.functionals.prediction.utils.size_finder import find_a_max_patch_shape, find_patch_and_halo_shapes
from plantseg.functionals.prediction.utils.slice_builder import SliceBuilder
from plantseg.functionals.prediction.utils.utils import get_stride_shape
from plantseg.training.augs import get_test_augmentations

logger = logging.getLogger(__name__)


def biio_prediction(
    raw: np.ndarray,
    input_layout: ImageLayout,
    model_id: str,
) -> np.ndarray:
    assert isinstance(input_layout, str)

    model = load_model_description(model_id)
    if isinstance(model, v0_4.ModelDescr):
        input_ids = [input_tensor.name for input_tensor in model.inputs]
    elif isinstance(model, v0_5.ModelDescr):
        input_ids = [input_tensor.id for input_tensor in model.inputs]
    else:
        assert_never(model)

    logger.info(f"Model expects these inputs: {input_ids}.")
    if len(input_ids) < 1:
        logger.error("Model needs no input tensor. PlantSeg does not support this yet.")
    if len(input_ids) > 1:
        logger.error("Model needs more than one input tensor. PlantSeg does not support this yet.")

    tensor_id = input_ids[0]
    axes = model.inputs[0].axes  # PlantSeg only supports one input tensor for now
    dims = tuple(
        AxisId('channel') if item.lower() == 'c' else AxisId(item.lower()) for item in input_layout
    )  # `AxisId` has to be "channel" not "c"
    members = {
        TensorId(tensor_id): Tensor(array=raw, dims=dims).transpose(
            [AxisId(a) if isinstance(a, str) else a.id for a in axes]
        )
    }
    sample = Sample(members=members, stat={}, id="raw")

    for a in axes:
        if isinstance(a, str):
            raise ValueError(f"Model has a string axis: {a}, please report issue to PlantSeg developers.")
    sizes_in_rdf = {a.id: a.size for a in axes}
    assert 'x' in sizes_in_rdf, "Model does not have 'x' axis in input tensor."
    size_to_check = sizes_in_rdf[AxisId('x')]
    if isinstance(size_to_check, int):  # e.g. 'emotional-cricket'
        # 'emotional-cricket' has {'batch': None, 'channel': 1, 'z': 100, 'y': 128, 'x': 128}
        input_block_shape = {
            TensorId(tensor_id): {
                a.id: a.size if isinstance(a.size, int) else 1
                for a in axes
                if not isinstance(a, str)  # for a.size/a.id type checking only
            }
        }
        sample_out = predict(model=model, inputs=sample, input_block_shape=input_block_shape)
    elif isinstance(size_to_check, v0_5.ParameterizedSize):  # e.g. 'philosophical-panda'
        # 'philosophical-panda' has:
        #   {'z': ParameterizedSize(min=1, step=1),
        #    'channel': 2,
        #    'y': ParameterizedSize(min=16, step=16),
        #    'x': ParameterizedSize(min=16, step=16)}
        blocksize_parameter = {
            (TensorId(tensor_id), a.id): (
                (96 - a.size.min) // a.size.step if isinstance(a.size, v0_5.ParameterizedSize) else 1
            )
            for a in axes
            if not isinstance(a, str)  # for a.size/a.id type checking only
        }
        sample_out = predict(model=model, inputs=sample, blocksize_parameter=blocksize_parameter)
    else:
        assert_never(size_to_check)

    assert isinstance(sample_out, Sample)
    if len(sample_out.members) != 1:
        logger.warning("Model has more than one output tensor. PlantSeg does not support this yet.")
    t = {i: o.transpose(['batch', 'channel', 'z', 'y', 'x']) for i, o in sample_out.members.items()}
    pmaps = []
    for i, bczyx in t.items():
        for czyx in bczyx:
            for zyx in czyx:
                pmaps.append(zyx.data.to_numpy())
    return pmaps  # FIXME: Wrong return type


def unet_prediction(
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
    config_path: Path | None = None,
    model_weights_path: Path | None = None,
) -> np.ndarray:
    """Generate prediction from raw data using a specified 3D U-Net model.

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
        if True:  # NOTE: For now, do not use native pytorch-3dunet prediction if using BioImage.IO models
            return biio_prediction(raw=raw, input_layout=input_layout, model_id=model_id)
        else:
            model, model_config, model_path = model_zoo.get_model_by_id(model_id)
    elif model_name is not None:  # PlantSeg zoo mode
        logger.info("Zoo prediction: Running model from PlantSeg official zoo.")
        model, model_config, model_path = model_zoo.get_model_by_name(model_name, model_update=model_update)
    else:
        raise ValueError("Either `model_name` or `model_id` or `model_path` must be provided.")
    state = torch.load(model_path, map_location="cpu", weights_only=True)

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
        maximum_patch_shape = find_a_max_patch_shape(model, model_config["in_channels"], device)
        raw_shape = raw.shape if input_layout == 'ZYX' else (1,) + raw.shape
        assert len(raw_shape) == 3
        patch, patch_halo = find_patch_and_halo_shapes(raw_shape, maximum_patch_shape, patch_halo, both_sides=False)

    logger.info(f"For raw in shape {raw.shape}: set patch shape {patch}, set halo shape {patch_halo}")

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
