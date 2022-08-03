import h5py
from pytorch3dunet.unet3d import utils
from plantseg.predictions.utils import get_loader_config, get_model_config, get_predictor_config, set_device
from plantseg.pipeline import gui_logger
from plantseg.io.io import load_shape
from plantseg.pipeline.steps import GenericPipelineStep


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
    def __init__(self,
                 input_paths,
                 model_name: str,
                 patch=(80, 160, 160),
                 stride='Accurate (slowest)',
                 device='cuda',
                 version='best',
                 model_update=False,
                 mirror_padding=(16, 32, 32),
                 input_type="data_float32",
                 output_type="data_float32",
                 out_ext=".h5",
                 state=True):
        h5_output_key = "predictions"

        valid_paths = _check_patch_size(input_paths, patch_size=patch)

        model, model_config, model_path = get_model_config(model_name, model_update=model_update)
        utils.load_checkpoint(model_path, model)

        device = set_device(device)
        model = model.to(device)

        predictor, predictor_config = get_predictor_config(model_name)
        self.predictor = predictor(model=model, config=model_config, device=device, **predictor_config)

        self.loader, self.loader_config = get_loader_config(model_name,
                                                            patch=patch,
                                                            stride=stride,
                                                            mirror_padding=mirror_padding)

        super().__init__(valid_paths,
                         input_type=input_type,
                         output_type=output_type,
                         save_directory=model_name,
                         out_ext=out_ext,
                         state=state,
                         file_suffix='_predictions',
                         h5_output_key=h5_output_key)

    def process(self, raw):
        raw_loader = self.loader(raw, **self.loader_config)
        pmaps = self.predictor(raw_loader)
        return pmaps[0]
