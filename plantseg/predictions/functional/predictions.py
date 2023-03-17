from typing import Tuple, Union

import numpy as np
from pytorch3dunet.unet3d import utils

from plantseg.dataprocessing.functional.dataprocessing import fix_input_shape
from plantseg.predictions.functional.utils import get_dataset_config, get_model_config, get_predictor_config, set_device


def unet_predictions(raw: np.array,
                     model_name: str,
                     patch: Tuple[int, int, int] = (80, 160, 160),
                     stride: Union[str, Tuple[int, int, int]] = 'Accurate (slowest)',
                     device: str = 'cuda',
                     version: str = 'best',
                     model_update: bool = False,
                     mirror_padding: Tuple[int, int, int] = (16, 32, 32)):
    """
    Predict boundaries predictions from raw data using a 3D U-Net model.

    Args:
        raw (np.array): raw data, must be a 3D array of shape (Z, Y, X) normalized between 0 and 1.
        model_name (str): name of the model to use. A complete list of available models can be found here:
        patch (tuple[int, int, int], optional): patch size to use for prediction. Defaults to (80, 160, 160).
        stride (Union[str, tuple[int, int, int]], optional): stride to use for prediction.
            If stride is defined as a string must be one of ['Accurate (slowest)', 'Balanced', 'Draft (fastest)'].
            Defaults to 'Accurate (slowest)'.
        device: (str, optional): device to use for prediction. Must be one of ['cpu', 'cuda', 'cuda:1', etc.].
            Defaults to 'cuda'.
        version (str, optional): version of the model to use, must be either 'best' or 'last'. Defaults to 'best'.
        model_update (bool, optional): if True will update the model to the latest version. Defaults to False.
        mirror_padding (tuple[int, int, int], optional): padding to use for prediction. Defaults to (16, 32, 32).

    Returns:
        np.array: predictions, 3D array of shape (Z, Y, X) with values between 0 and 1.

    """
    model, model_config, model_path = get_model_config(model_name, model_update=model_update, version=version)
    utils.load_checkpoint(model_path, model)

    device = set_device(device)
    model = model.to(device)

    predictor, predictor_config = get_predictor_config(model_name)
    predictor = predictor(model=model,
                          config=model_config,
                          device=device,
                          verbose_logging=False, **predictor_config)

    dataset_builder, dataset_config = get_dataset_config(model_name,
                                                         patch=patch,
                                                         stride=stride,
                                                         mirror_padding=mirror_padding)

    raw = fix_input_shape(raw)
    raw = raw.astype('float32')

    raw_loader = dataset_builder(raw, verbose_logging=False, **dataset_config)
    pmaps = predictor(raw_loader)
    pmaps = fix_input_shape(pmaps[0])
    return pmaps
