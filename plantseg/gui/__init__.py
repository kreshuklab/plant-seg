import os
import tkinter

import yaml

from plantseg import plantseg_global_path, custom_zoo, model_zoo_path

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


def list_models():
    """ list model zoo """
    zoo_config = os.path.join(model_zoo_path)
    zoo_config = yaml.load(open(zoo_config, 'r'),
                             Loader=yaml.FullLoader)

    custom_zoo_config = yaml.load(open(custom_zoo, 'r'),
                             Loader=yaml.FullLoader)

    if custom_zoo_config is None:
        custom_zoo_config = {}

    zoo_config.update(custom_zoo_config)
    models = list(zoo_config.keys())
    return models