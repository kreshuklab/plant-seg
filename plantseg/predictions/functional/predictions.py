from typing import Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from pytorch3dunet.unet3d import utils
from plantseg.dataprocessing.functional.dataprocessing import fix_input_shape
from plantseg.predictions.utils import get_dataset_config, get_model_config, get_predictor_config, set_device


def unet_predictions(raw: ArrayLike,
                     model_name: str,
                     patch: Tuple[int, int, int] = (80, 160, 160),
                     stride: Union[str, Tuple[int, int, int]] = 'Accurate (slowest)',
                     device: str = 'cuda',
                     version: str = 'best',
                     model_update: bool = False,
                     mirror_padding: Tuple[int, int, int] = (16, 32, 32)):
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
