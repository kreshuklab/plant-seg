import gc

import numpy as np
import torch
from torch import nn

from plantseg.io.io import load_shape
from plantseg.pipeline import gui_logger
from plantseg.pipeline.steps import GenericPipelineStep
from plantseg.predictions.functional.array_predictor import ArrayPredictor
from plantseg.predictions.functional.utils import get_array_dataset, get_model_config, get_patch_halo, find_batch_size


def _check_patch_size(paths, patch_size):
    axis = ['z', 'x', 'y']
    valid_paths = []

    for path in paths:
        incorrect_axis = []
        raw_size = load_shape(path, key='raw')

        for _ax, _patch_size, _raw_size in zip(axis, patch_size, raw_size):
            if _patch_size > _raw_size:
                incorrect_axis.append(_ax)

        if len(incorrect_axis) > 0:
            gui_logger.warning(f"Incorrect Patch size for {path}.\n Patch size {patch_size} along {incorrect_axis}"
                               f" axis (axis order zxy) is too big for an image of size {raw_size},"
                               f" patch size should be smaller or equal than the raw stack size. \n"
                               f"{path} will be skipped.")
        else:
            valid_paths.append(path)

    if len(valid_paths) == 0:
        raise RuntimeError(f"No valid path found for the patch size specified in the PlantSeg config. \n"
                           f" Patch size should be smaller or equal than the raw stack size.")
    return valid_paths


class UnetPredictions(GenericPipelineStep):
    def __init__(self, input_paths, model_name, patch=(80, 160, 160), stride_ratio=0.75, device='cuda',
                 model_update=False, input_type="data_float32", output_type="data_float32", out_ext=".h5", state=True):
        self.patch = patch
        self.model_name = model_name
        self.stride_ratio = stride_ratio

        h5_output_key = "predictions"
        valid_paths = _check_patch_size(input_paths, patch_size=patch) if state else input_paths

        super().__init__(valid_paths,
                         input_type=input_type,
                         output_type=output_type,
                         save_directory=model_name,
                         out_ext=out_ext,
                         state=state,
                         file_suffix='_predictions',
                         h5_output_key=h5_output_key)

        model, model_config, model_path = get_model_config(model_name, model_update=model_update)
        state = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state)

        patch_halo = get_patch_halo(model_name)
        batch_size = find_batch_size(model, model_config['in_channels'], patch, patch_halo, device)
        gui_logger.info(f'Using batch size of {batch_size} for prediction')

        if torch.cuda.device_count() > 1 and device != 'cpu':
            model = nn.DataParallel(model)
            gui_logger.info(f'Using {torch.cuda.device_count()} GPUs for prediction')

        model = model.to(device)

        self.predictor = ArrayPredictor(model=model, batch_size=batch_size, out_channels=model_config['out_channels'],
                                        device=device, patch_halo=patch_halo)

    def process(self, raw: np.ndarray) -> np.ndarray:
        dataset = get_array_dataset(raw, self.model_name, patch=self.patch, stride_ratio=self.stride_ratio)
        pmaps = self.predictor(dataset)
        return pmaps[0]
