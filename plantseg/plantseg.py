import glob
import os
import argparse
import yaml
import h5py

from .models.checkmodels import check_models


def _read_path(config):
    if os.path.isdir(config["path"]):
        path = os.path.join(config["path"], "*")
    else:
        path, ext = os.path.splitext(config["path"])
        path = f"{path}*{ext}"
    paths = glob.glob(path)
    only_file = []
    for path in paths:
        if os.path.isfile(path):
            only_file.append(path)
    return sorted(only_file)


def _generate_new_paths(all_paths, new_name, suffix=''):
    all_paths_new = []
    for path in all_paths:
        basepath, basename = os.path.split(path)
        basename = f"{os.path.splitext(basename)[0]}{suffix}.h5"
        all_paths_new.append(os.path.join(basepath, new_name, basename))
    return all_paths_new


def _create_dir_structure(file_path, preprocessing_name='', model_name='', seg_name=''):
    dir_path = os.path.dirname(file_path)
    dir_path = os.path.join(dir_path, preprocessing_name, model_name, seg_name)
    os.makedirs(dir_path, exist_ok=True)


def _load_config():
    parser = argparse.ArgumentParser(description='Plant cell instance segmentation script')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    return config


def _import_preprocessing_pipeline(_config, all_paths):
    from .dataprocessing.dataprocessing import DataPreProcessing3D
    processing = DataPreProcessing3D(_config, all_paths)
    return processing


def _import_postprocessing_pipeline(_config, all_paths, dataset):
    from .dataprocessing.dataprocessing import DataPostProcessing3D
    processing = DataPostProcessing3D(_config, all_paths, dataset)
    return processing


def _create_predict_config(_config, all_paths):
    """ Creates the configuration file needed for running the neural network inference"""

    # Load template config
    import torch
    file_dir = os.path.dirname(os.path.abspath(__file__))
    config = yaml.load(open(os.path.join(file_dir, "predictions", "config_predict_template.yaml"), 'r'),
                       Loader=yaml.FullLoader)

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
    config["model_path"] = os.path.join(home,
                                        ".plantseg_models",
                                        _config['model_name'],
                                        f"{_config['version']}_checkpoint.pytorch")

    # Load train config and add missing info
    config_train = yaml.load(open(os.path.join(home,
                                        ".plantseg_models",
                                        _config['model_name'],
                                        "config_train.yml"), 'r'),
                             Loader=yaml.FullLoader)
    #
    for key, value in config_train["model"].items():
        config["model"][key] = value

    config["model_name"] = _config["model_name"]
    return config


def _import_predction_pipeline(_config, all_paths):
    from .predictions.predict import ModelPredictions
    config = _create_predict_config(_config, all_paths)
    model_predictions = ModelPredictions(config)
    return model_predictions


def _import_segmentation_algorithm(config, predictions_paths):
    name = config["name"]

    if name == "GASP" or name == "MutexWS":
        from .segmentation.gasp import GaspFromPmaps as Segmentation

    elif name == "DtWatershed":
        from .segmentation.watershed import DtWatershedFromPmaps as Segmentation

    elif name == "MultiCut":
        from .segmentation.multicut import MulticutFromPmaps as Segmentation

    elif name == "RandomWalker":
        from .segmentation.randomwalker import DtRandomWalkerFromPmaps as Segmentation

    else:
        raise NotImplementedError

    segmentation = Segmentation(predictions_paths)
    for name in segmentation.__dict__.keys():
        if name in config:
            segmentation.__dict__[name] = config[name]

    return segmentation


def _load_file(path, dataset):
    with h5py.File(path, "r") as f:
        data = f[dataset][...]

    return data


class dummy:
    def __init__(self, phase):
        self.phase = phase

    def __call__(self,):
        print(f"Skipping {self.phase}: Nothing to do")


def raw2seg(config):
    # read files
    all_paths_raw = _read_path(config)

    # creates predictions paths

    # Create directory structure for segmentation results
    if "preprocessing" in config:
        if "save_directory" in config["preprocessing"]:
            preprocessing_save_directory = config["preprocessing"]["save_directory"]
        else:
            preprocessing_save_directory = ""
    else:
        preprocessing_save_directory = ""

    if "unet_prediction" in config:
        if "model_name" in config["unet_prediction"]:
            unet_save_directory = config["unet_prediction"]["model_name"]
        else:
            unet_save_directory = ""
    else:
        unet_save_directory = ""

    if "segmentation" in config:
        if "save_directory" in config["segmentation"]:
            segmentation_save_directory = config["segmentation"]["save_directory"]
        else:
            segmentation_save_directory = ""
    else:
        segmentation_save_directory = ""

    [_create_dir_structure(file_path,
                           preprocessing_save_directory,
                           unet_save_directory,
                           segmentation_save_directory) for file_path in all_paths_raw]

    if 'preprocessing' in config.keys() and config['preprocessing']['state']:
        # creates segmentation processed paths
        all_paths_processed = _generate_new_paths(all_paths_raw, config["preprocessing"]["save_directory"])
        preprocessing = _import_preprocessing_pipeline(config["preprocessing"], all_paths_raw)
    else:
        all_paths_processed = all_paths_raw
        preprocessing = dummy("prepocessing")

    # Import predictions pipeline
    if 'unet_prediction' in config.keys() and config['unet_prediction']['state']:
        all_paths_predicted = _generate_new_paths(all_paths_processed, config["unet_prediction"]["model_name"],
                                                  suffix="_predictions")
        predictions = _import_predction_pipeline(config["unet_prediction"], all_paths_processed)
        if "postprocessing" in config["unet_prediction"].keys() and\
                config['unet_prediction']['postprocessing']['state']:
            predictions_postprocessing = _import_postprocessing_pipeline(config["unet_prediction"]["postprocessing"],
                                                                         all_paths_predicted, "predictions")
        else:
            predictions_postprocessing = dummy("predictions postprocessing")

    else:
        all_paths_predicted = all_paths_processed
        predictions = dummy("predictions")
        predictions_postprocessing = dummy("predictions postprocessing")

    # Import segmentation pipeline
    if "segmentation" in config.keys() and config['segmentation']['state']:
        all_paths_segmented = _generate_new_paths(all_paths_predicted, config["segmentation"]["save_directory"],
                                                  suffix=f"_{config['segmentation']['save_directory']}".lower())
        segmentation = _import_segmentation_algorithm(config["segmentation"], all_paths_predicted)
        print("Segmentation Pipeline Initialized - Params:", segmentation.__dict__)
        if "postprocessing" in config["unet_prediction"].keys() and \
                config['segmentation']['postprocessing']['state']:
            segmentation_postprocessing = _import_postprocessing_pipeline(config["segmentation"]["postprocessing"],
                                                                          all_paths_segmented, "segmentation")
        else:
            segmentation_postprocessing = dummy("predictions postprocessing")

    else:
        segmentation = dummy("segmentation")
        segmentation_postprocessing = dummy("segmentation postprocessing")

    # Run pipelines
    print("Inference start")
    preprocessing()

    print("Starting Predictions")
    predictions()
    print("Predictions Done")
    predictions_postprocessing()

    print("Starting Segmentation")
    segmentation()
    segmentation_postprocessing()
    print("All Done!")


if __name__ == "__main__":
    # Load general configuration file
    config = _load_config()
    raw2seg(config)
