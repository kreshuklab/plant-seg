import glob
import os
import argparse
import yaml
import h5py
from models.checkmodels import check_models


def _read_path(config):
    path = os.path.splitext(config["path"])[0]
    return sorted(glob.glob(path + "*.h5"))


def _create_dir_structure(file_path, model_name='', seg_name=''):
    dir_path = os.path.dirname(file_path)
    dir_path = dir_path + '/' + model_name + '/' + seg_name + '/'
    os.makedirs(dir_path, exist_ok=True)


def _load_config():
    parser = argparse.ArgumentParser(description='Plant cell instance segmentation script')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    return config


def _import_preprocessing_pipeline(_config):
    from dataprocessing.dataprocessing import DataPreProcessing
    processing = DataPreProcessing(_config)
    return processing


def _import_postprocessing_pipeline(_config):
    from dataprocessing.dataprocessing import DataPostProcessing
    processing = DataPostProcessing(_config)
    return processing


def _create_predict_config(_config, all_paths):
    """ Creates the configuration file needed for running the neural network inference"""

    # Load template config
    import torch
    config = yaml.load(open("./predictions/config_predict_template.yaml", 'r'), Loader=yaml.FullLoader)

    # Add patch and stride size
    if "patch" in _config.keys():
        config["datasets"]["patch"] = _config["patch"]
    if "stride" in _config.keys():
        config["datasets"]["stride"] = _config["stride"]

    # Add paths to raw data
    config["datasets"]["test_path"] = all_paths

    # Add correct device for inference
    if _config["device"] == 'cuda':
        config["device"] = torch.device("cuda:0")
    elif _config["device"] == 'cpu':
        config["device"] = torch.device("cpu")
    else:
        raise NotImplementedError

    # check if all files are in the data directory (~/.plantseg_models/)
    check_models(_config['model_name'], update_files=_config['model_update'])

    # Add model path
    home = os.path.expanduser("~")
    config["model_path"] = f"{home}/.plantseg_models/'{_config['model_name']}'/'{_config['version']}'_checkpoint.pytorch"

    # Load train config and add missing info
    config_train = yaml.load(open(f"{home}/.plantseg_models/{_config['model_name']}/config_train.yml", 'r'),
                             Loader=yaml.FullLoader)
    #
    for key, value in config_train["model"].items():
        config["model"][key] = value
    return config


def _import_predction_pipeline(_config, all_paths):
    from predictions.predict import ModelPredictions
    config = _create_predict_config(_config, all_paths)
    model_predictions = ModelPredictions(config)
    return model_predictions


def _import_segmentation_algorithm(config):
    name = config["name"]

    if name == "GASP" or name == "MutexWS":
        from segmentation.gasp import GaspFromPmaps as Segmentation

    elif name == "DtWatershed":
        from segmentation.watershed import DtWatershedFromPmaps as Segmentation

    elif name == "MultiCut":
        from segmentation.multicut import MulticutFromPmaps as Segmentation

    elif name == "RandomWalker":
        from segmentation.randomwalker import DtRandomWalkerFromPmaps as Segmentation

    else:
        raise NotImplementedError

    segmentation = Segmentation()
    for name in segmentation.__dict__.keys():
        if name in config:
            segmentation.__dict__[name] = config[name]

    return segmentation


def _load_file(path, dataset):
    with h5py.File(path, "r") as f:
        data = f[dataset][...]

    return data


def dummy(phase):
    print(f"Skipping {phase}: Nothing to do")
    return 0


def raw2seg():
    # Load general configuration file
    config = _load_config()

    # read files
    all_paths = _read_path(config)

    if 'preprocessing' in config.keys():
        preprocessing = _import_preprocessing_pipeline(config["preprocessing"])
    else:
        preprocessing = dummy("prepocessing")

    # Import predictions pipeline
    if 'unet_predictions' in config.keys():
        predictions = _import_predction_pipeline(config["unet_prediction"], all_paths)
        if config["postprocessing"] in config["unet_prediction"].keys():
            predictions_postprocessing = _import_postprocessing_pipeline(config["unet_prediction"]["postprocessing"])
    else:
        predictions = dummy("predictions")
        segmentation_postprocessing = dummy("predictions postprocessing")

    # Import segmentation pipeline
    if "segmentation" in config.keys():
        segmentation = _import_segmentation_algorithm(config["segmentation"])
        print("Segmentation Pipeline Initialized - Params:", segmentation.__dict__)
        if config["postprocessing"] in config["unet_prediction"].keys():
            segmentation_postprocessing = _import_postprocessing_pipeline(config["segmentation"]["postprocessing"])

    else:
        segmentation = dummy("segmentation")
        segmentation_postprocessing = dummy("segmentation postprocessing")

    # Create directory structure for segmentation results
    [_create_dir_structure(file_path, config["unet_prediction"]["model_name"], config["segmentation"]["name"]) for file_path in all_paths]

    # Run pipelines
    print("Inference start")
    preprocessing()
    predictions()
    predictions_postprocessing()
    segmentation()
    segmentation_postprocessing()


if __name__ == "__main__":
    raw2seg()
