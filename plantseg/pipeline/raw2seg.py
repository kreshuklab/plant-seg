import numpy as np

from plantseg.dataprocessing.dataprocessing import DataPostProcessing3D
from plantseg.dataprocessing.dataprocessing import DataPreProcessing3D
from plantseg.io.io import load_shape
from plantseg.pipeline import gui_logger
from plantseg.pipeline.config_validation import config_validation
from plantseg.pipeline.utils import load_paths
from plantseg.predictions.predict import UnetPredictions
from plantseg.segmentation.utils import configure_segmentation_step


def configure_preprocessing_step(input_paths, config):
    output_type = config.get('output_type', "data_uint8")
    save_directory = config.get('save_directory', 'PreProcessing')
    factor = config.get('factor', [1, 1, 1])

    filter_type = None
    filter_param = None
    if config["filter"]["state"]:
        filter_type = config["filter"]["type"]
        filter_param = config["filter"]["filter_param"]

    state = config.get('state', True)
    crop = config.get('crop_volume', None)
    return DataPreProcessing3D(input_paths, input_type="data_float32", output_type=output_type,
                               save_directory=save_directory, factor=factor, filter_type=filter_type,
                               filter_param=filter_param, state=state, crop=crop)


def configure_cnn_step(input_paths, config):
    model_name = config['model_name']
    patch = config.get('patch', (80, 160, 160))
    stride_ratio = config.get('stride_ratio', 0.75)
    device = config.get('device', 'cuda')
    state = config.get('state', True)
    model_update = config.get('model_update', False)
    return UnetPredictions(input_paths, model_name=model_name, patch=patch, stride_ratio=stride_ratio,
                           device=device, model_update=model_update, state=state)


def configure_cnn_postprocessing_step(input_paths, config):
    return _create_postprocessing_step(input_paths, input_type="data_float32", config=config)


def configure_segmentation_postprocessing_step(input_paths, config):
    return _create_postprocessing_step(input_paths, input_type="labels", config=config)


def _create_postprocessing_step(input_paths, input_type, config):
    output_type = config.get('output_type', input_type)
    save_directory = config.get('save_directory', 'PostProcessing')
    factor = config.get('factor', [1, 1, 1])
    output_shapes = config.get('output_shapes', None)
    out_ext = ".tiff" if config["tiff"] else ".h5"
    state = config.get('state', True)
    save_raw = config.get('save_raw', False)
    return DataPostProcessing3D(input_paths, input_type=input_type, output_type=output_type,
                                save_directory=save_directory, factor=factor, out_ext=out_ext,
                                state=state, save_raw=save_raw, output_shapes=output_shapes)


def _validate_cnn_postprocessing_rescaling(input_paths, config):
    input_shapes = [load_shape(input_path) for input_path in input_paths]
    # if CNN output was rescaled, it needs to be scaled back to the correct input size
    if not np.array_equal(np.array([1, 1, 1]), config["cnn_postprocessing"]["factor"]):
        # prevent scipy zoom rounding errors
        config["cnn_postprocessing"]["output_shapes"] = input_shapes


def raw2seg(config):
    config = config_validation(config)

    input_paths = load_paths(config["path"])
    gui_logger.info(f"Running the pipeline on: {input_paths}")

    gui_logger.info("Executing pipeline, see terminal for verbose logs.")
    all_pipeline_steps = [
        ('preprocessing', configure_preprocessing_step),
        ('cnn_prediction', configure_cnn_step),
        ('cnn_postprocessing', configure_cnn_postprocessing_step),
        ('segmentation', configure_segmentation_step),
        ('segmentation_postprocessing', configure_segmentation_postprocessing_step)
    ]

    for pipeline_step_name, pipeline_step_setup in all_pipeline_steps:
        if pipeline_step_name == 'preprocessing':
            _validate_cnn_postprocessing_rescaling(input_paths, config)

        gui_logger.info(
            f"Executing pipeline step: '{pipeline_step_name}'. Parameters: '{config[pipeline_step_name]}'. Files {input_paths}.")
        pipeline_step = pipeline_step_setup(input_paths, config[pipeline_step_name])
        output_paths = pipeline_step()

        # replace input_paths for all pipeline steps except DataPostProcessing3D
        if not isinstance(pipeline_step, DataPostProcessing3D):
            input_paths = output_paths

    gui_logger.info(f"Pipeline execution finished!")
