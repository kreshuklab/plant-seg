import glob
import os
import tkinter
from pathlib import Path
from shutil import copy2
from typing import Tuple

import yaml

from plantseg import custom_zoo, home_path, PLANTSEG_MODELS_DIR
from plantseg import model_zoo_path

stick_all = tkinter.N + tkinter.S + tkinter.E + tkinter.W
stick_ew = tkinter.E + tkinter.W
stick_new = tkinter.N + tkinter.E + tkinter.W
PLANTSEG_GREEN = (208, 240, 192)


def var_to_tkinter(var):
    """ transform python variables in tkinter variables"""
    if isinstance(var, bool):
        tk_var = tkinter.BooleanVar()

    elif isinstance(var, str):
        tk_var = tkinter.StringVar()

    elif isinstance(var, int):
        tk_var = tkinter.IntVar()

    elif isinstance(var, float):
        tk_var = tkinter.DoubleVar()

    elif isinstance(var, list):
        tk_var = tkinter.StringVar()

    tk_var.set(var)
    return tk_var


def convert_rgb(rgb=(0, 0, 0)):
    """ rgb to tkinter friendly format"""
    rgb = tuple(rgb)
    return "#%02x%02x%02x" % rgb


def get_model_config():
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
    zoo_config = get_model_config()
    models = list(zoo_config.keys())
    return models


def get_model_resolution(model):
    """ list model zoo """
    zoo_config = get_model_config()
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
    custom_zoo_dict[new_model_name]["path"] = location
    custom_zoo_dict[new_model_name]["resolution"] = resolution
    custom_zoo_dict[new_model_name]["description"] = description

    with open(custom_zoo, 'w') as f:
        yaml.dump(custom_zoo_dict, f)

    return True, None
