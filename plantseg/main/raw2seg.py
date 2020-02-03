from plantseg.main import load_paths, dummy


def import_preprocessing_pipeline(input_paths, _config):
    from ..dataprocessing.dataprocessing import DataPreProcessing3D
    processing = DataPreProcessing3D(input_paths, _config)
    return processing


def import_cnn_pipeline(input_paths, _config):
    from plantseg.predictions import create_predict_config
    from ..predictions.predict import ModelPredictions
    cnn_config = create_predict_config(input_paths, _config)
    model_predictions = ModelPredictions(cnn_config)
    return model_predictions


def import_cnn_postprocessing_pipeline(input_paths, _config):
    from ..dataprocessing.dataprocessing import DataPostProcessing3D
    processing = DataPostProcessing3D(input_paths, _config, data_type="data_float32")
    return processing


def import_segmentation_pipeline(input_paths, _config):
    from plantseg.segmentation import configure_segmentation
    segmentation = configure_segmentation(input_paths, _config)
    return segmentation


def import_segmentation_postprocessing_pipeline(input_paths, _config):
    from ..dataprocessing.dataprocessing import DataPostProcessing3D
    processing = DataPostProcessing3D(input_paths, _config, data_type="labels")
    return processing


class SetupProcess:
    def __init__(self, paths, config, pipeline_name, import_function):
        if pipeline_name in config.keys() and config[pipeline_name]['state']:
            # Import pipeline and assign paths
            self.pipeline = import_function(paths, config[pipeline_name])
        else:
            # If pipeline is not configured or state=False use dummy
            self.pipeline = dummy(paths, pipeline_name)

    def __call__(self):
        return self.pipeline()


def raw2seg(config):
    # read files
    print("File paths loading...")
    paths = load_paths(config)

    print("\nStart processing... see terminal for verbose logs")
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

    print("All done! \n")
