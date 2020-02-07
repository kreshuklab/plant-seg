from plantseg.main import gui_logger
from plantseg.main.utils import load_paths
from plantseg.pipeline.steps import PlaceholderPipelineStep
from plantseg.predictions.utils import create_predict_config
from plantseg.segmentation.utils import configure_segmentation
from ..dataprocessing.dataprocessing import DataPostProcessing3D
from ..dataprocessing.dataprocessing import DataPreProcessing3D
from ..predictions.predict import ModelPredictions


def import_preprocessing_pipeline(input_paths, _config):
    output_type = _config.get('output_type', "data_uint8")
    save_directory = _config.get('save_directory', 'PreProcessing')
    factor = _config.get('factor', [1, 1, 1])

    filter_type = None
    filter_param = None
    if _config["filter"]["state"]:
        filter_type = _config["filter"]["type"]
        filter_param = _config["filter"]["param"]

    return DataPreProcessing3D(input_paths, input_type="data_float32", output_type=output_type,
                               save_directory=save_directory, factor=factor, filter_type=filter_type,
                               filter_param=filter_param)


def import_cnn_pipeline(input_paths, _config):
    cnn_config = create_predict_config(input_paths, _config)
    model_predictions = ModelPredictions(cnn_config)
    return model_predictions


def import_cnn_postprocessing_pipeline(input_paths, _config):
    return _create_postprocessing_step(input_paths, input_type="data_float32", config=_config)


def import_segmentation_pipeline(input_paths, _config):
    segmentation = configure_segmentation(input_paths, _config)
    return segmentation


def import_segmentation_postprocessing_pipeline(input_paths, _config):
    return _create_postprocessing_step(input_paths, input_type="labels", config=_config)


def _create_postprocessing_step(input_paths, input_type, config):
    output_type = config.get('output_type', input_type)
    save_directory = config.get('save_directory', 'PostProcessing')
    factor = config.get('factor', [1, 1, 1])
    out_ext = ".tiff" if config["tiff"] else ".h5"
    return DataPostProcessing3D(input_paths, input_type=input_type, output_type=output_type,
                                save_directory=save_directory, factor=factor, out_ext=out_ext)


class SetupProcess:
    def __init__(self, paths, config, pipeline_name, import_function):
        if pipeline_name in config.keys() and config[pipeline_name]['state']:
            gui_logger.info(
                f"Executing pipeline step: '{pipeline_name}' with parameters: '{config[pipeline_name]}' for the following files: '{paths}")
            # Import pipeline and assign paths
            self.pipeline = import_function(paths, config[pipeline_name])
        else:
            gui_logger.info(f"Skipping pipeline step: '{pipeline_name}'. Nothing do to here.")
            # If pipeline is not configured or state=False use PlaceholderStep
            self.pipeline = PlaceholderPipelineStep(paths, pipeline_name)

    def __call__(self):
        return self.pipeline()


def raw2seg(config):
    paths = load_paths(config)
    gui_logger.info(f"Running the pipeline on: {paths}")

    gui_logger.info("Executing pipeline, see terminal for verbose logs.")
    all_pipelines = [('preprocessing', import_preprocessing_pipeline),
                     ('cnn_prediction', import_cnn_pipeline),
                     ('cnn_postprocessing', import_cnn_postprocessing_pipeline),
                     ('segmentation', import_segmentation_pipeline),
                     ('segmentation_postprocessing', import_segmentation_postprocessing_pipeline)]

    for pipeline_name, import_pipeline in all_pipelines:
        # setup generic pipeline
        pipeline = SetupProcess(paths, config, pipeline_name, import_pipeline)
        # run pipeline and update paths
        paths = pipeline()

    gui_logger.info(f"Pipeline execution finished! See the results in {paths}")
