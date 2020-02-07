from plantseg.dataprocessing.dataprocessing import DataPostProcessing3D
from plantseg.dataprocessing.dataprocessing import DataPreProcessing3D
from plantseg.pipeline import gui_logger
from plantseg.pipeline.utils import load_paths
from plantseg.predictions.predict import UnetPredictions
from plantseg.predictions.utils import create_predict_config
from plantseg.segmentation.utils import configure_segmentation_step


def configure_preprocessing_step(input_paths, config):
    output_type = config.get('output_type', "data_uint8")
    save_directory = config.get('save_directory', 'PreProcessing')
    factor = config.get('factor', [1, 1, 1])

    filter_type = None
    filter_param = None
    if config["filter"]["state"]:
        filter_type = config["filter"]["type"]
        filter_param = config["filter"]["param"]

    state = config.get('state', True)
    return DataPreProcessing3D(input_paths, input_type="data_float32", output_type=output_type,
                               save_directory=save_directory, factor=factor, filter_type=filter_type,
                               filter_param=filter_param, state=state)


def configure_cnn_step(input_paths, config):
    cnn_config = create_predict_config(input_paths, config)
    return UnetPredictions(cnn_config)


def configure_cnn_postprocessing_step(input_paths, config):
    return _create_postprocessing_step(input_paths, input_type="data_float32", config=config)


def configure_segmentation_postprocessing_step(input_paths, config):
    return _create_postprocessing_step(input_paths, input_type="labels", config=config)


def _create_postprocessing_step(input_paths, input_type, config):
    output_type = config.get('output_type', input_type)
    save_directory = config.get('save_directory', 'PostProcessing')
    factor = config.get('factor', [1, 1, 1])
    out_ext = ".tiff" if config["tiff"] else ".h5"
    state = config.get('state', True)
    return DataPostProcessing3D(input_paths, input_type=input_type, output_type=output_type,
                                save_directory=save_directory, factor=factor, out_ext=out_ext, state=state)


def raw2seg(config):
    input_paths = load_paths(config)
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
        gui_logger.info(
            f"Executing pipeline step: '{pipeline_step_name}'. Parameters: '{config[pipeline_step_name]}'. Files {input_paths}.")
        pipeline_step = pipeline_step_setup(input_paths, config[pipeline_step_name])
        output_paths = pipeline_step()

        # replace input_paths for all pipeline steps except DataPostProcessing3D
        if not isinstance(pipeline_step, DataPostProcessing3D):
            input_paths = output_paths

    gui_logger.info(f"Pipeline execution finished! See the results in {input_paths}")
