import glob
import os
from pathlib import Path
from shutil import copy2
from typing import Tuple

import requests
import yaml

from plantseg import model_zoo_path, custom_zoo, home_path, PLANTSEG_MODELS_DIR, plantseg_global_path
from plantseg.pipeline import gui_logger


def get_model_zoo():
    zoo_config = os.path.join(model_zoo_path)
    zoo_config = yaml.load(open(zoo_config, 'r'),
                           Loader=yaml.FullLoader)

    custom_zoo_config = yaml.load(open(custom_zoo, 'r'),
                                  Loader=yaml.FullLoader)

    if custom_zoo_config is None:
        custom_zoo_config = {}

    zoo_config.update(custom_zoo_config)
    return zoo_config


def list_models():
    """ list model zoo """
    zoo_config = get_model_zoo()
    models = list(zoo_config.keys())
    return models


def get_model_resolution(model):
    """ list model zoo """
    zoo_config = get_model_zoo()
    resolution = zoo_config[model].get('resolution', [1., 1., 1.])
    return resolution


def add_custom_model(new_model_name: str,
                     location: Path = Path.home(),
                     resolution: Tuple[float, float, float] = (1., 1., 1.),
                     description: str = ''):

    dest_dir = os.path.join(home_path, PLANTSEG_MODELS_DIR, new_model_name)
    os.makedirs(dest_dir, exist_ok=True)
    all_files = glob.glob(os.path.join(location, "*"))
    all_expected_files = ['config_train.yml',
                          'last_checkpoint.pytorch',
                          'best_checkpoint.pytorch']
    for file in all_files:
        if os.path.basename(file) in all_expected_files:
            copy2(file, dest_dir)
            all_expected_files.remove(os.path.basename(file))

    if len(all_expected_files) != 0:
        msg = f'It was not possible to find in the directory specified {all_expected_files}, ' \
              f'the model can not be loaded.'
        return False, msg

    custom_zoo_dict = yaml.load(open(custom_zoo, 'r'), Loader=yaml.FullLoader)
    if custom_zoo_dict is None:
        custom_zoo_dict = {}

    custom_zoo_dict[new_model_name] = {}
    custom_zoo_dict[new_model_name]["path"] = str(location)
    custom_zoo_dict[new_model_name]["resolution"] = resolution
    custom_zoo_dict[new_model_name]["description"] = description

    with open(custom_zoo, 'w') as f:
        yaml.dump(custom_zoo_dict, f)

    return True, None


CONFIG_TRAIN_YAML = "config_train.yml"
BEST_MODEL_PYTORCH = "best_checkpoint.pytorch"
LAST_MODEL_PYTORCH = "last_checkpoint.pytorch"


def get_train_config(model_name, model_update=False):
    check_models(model_name, update_files=model_update)
    # Load train config and add missing info
    train_config_path = os.path.join(home_path,
                                     PLANTSEG_MODELS_DIR,
                                     model_name,
                                     CONFIG_TRAIN_YAML)
    with open(train_config_path, 'r') as f:
        config_train = yaml.full_load(f)
    return config_train


def download_model(url, out_dir='.'):
    for file in [CONFIG_TRAIN_YAML, BEST_MODEL_PYTORCH, LAST_MODEL_PYTORCH]:
        with requests.get(f'{url}{file}', allow_redirects=True) as r:
            with open(os.path.join(out_dir, file), 'wb') as f:
                f.write(r.content)


def check_models(model_name, update_files=False):
    """
    Simple script to check and download trained modules
    """
    if os.path.isdir(model_name):
        model_dir = model_name
    else:
        model_dir = os.path.join(os.path.expanduser("~"), PLANTSEG_MODELS_DIR, model_name)
        # Check if model directory exist if not create it
        if ~os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

    model_config_path = os.path.exists(os.path.join(model_dir, CONFIG_TRAIN_YAML))
    model_best_path = os.path.exists(os.path.join(model_dir, BEST_MODEL_PYTORCH))
    model_last_path = os.path.exists(os.path.join(model_dir, LAST_MODEL_PYTORCH))

    # Check if files are there, if not download them
    if (not model_config_path or
            not model_best_path or
            not model_last_path or
            update_files):

        # Read config
        model_file = os.path.join(plantseg_global_path, "resources", "models_zoo.yaml")
        config = yaml.load(open(model_file, 'r'), Loader=yaml.FullLoader)

        if model_name in config:
            url = config[model_name]["path"]

            gui_logger.info(f"Downloading model files from: '{url}' ...")
            download_model(url, out_dir=model_dir)
        else:
            raise RuntimeError(f"Custom model {model_name} corrupted. Required files not found.")
    return True
