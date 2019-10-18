import glob
import os
import argparse
import yaml
import h5py
#import torch
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


def _import_predction_pipeline(_config, all_paths):
    config = yaml.load(open("./predictions/config_predict_template.yaml", 'r'), Loader=yaml.FullLoader)

    if "patch" in _config.keys():
        config["datasets"]["patch"] = _config["patch"]
    if "stride" in _config.keys():
        config["datasets"]["stride"] = _config["stride"]

    config["datasets"]["test_path"] = all_paths

    if _config["device"] == 'cuda':
        config["device"] = 0 # torch.device("cuda:0")
    elif _config["device"] == 'cpu':
        config["device"] = 0 # torch.device("cpu")
    else:
        raise NotImplementedError

    check_models(_config['model_name'])

    home = os.path.expanduser("~")
    config["model_path"] = f"{home}/.plantseg_models/'{_config['model_name']}'/'{_config['version']}'_checkpoint.pytorch"

    config_train = yaml.load(open(f"{home}/.plantseg_models/{_config['model_name']}/config_train.yml", 'r'),
                             Loader=yaml.FullLoader)
    for key, value in config_train["model"].items():
        config["model"][key] = value

    print(config)


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


def raw2seg():
    config = _load_config()
    all_paths = _read_path(config)
    segmentation = _import_segmentation_algorithm(config["segmentation"])
    print("Segmentation Pipeline Initialized - Params:", segmentation.__dict__)

    predictions = _import_predction_pipeline(config["unet_prediction"], all_paths)


    for file_path in all_paths:
        _create_dir_structure(file_path, "predtest", config["segmentation"]["name"])

        break


def raw2pmaps():
    print("raw2pmaps")


def pmaps2seg():
    print("pmaps2seg")


if __name__ == "__main__":
    raw2seg()