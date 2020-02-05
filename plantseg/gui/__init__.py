import os
import tkinter

import yaml

from plantseg import plantseg_global_path

stick_all = tkinter.N + tkinter.S + tkinter.E + tkinter.W
stick_ew = tkinter.E + tkinter.W
stick_new = tkinter.N + tkinter.E + tkinter.W


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
    model_config = os.path.join(plantseg_global_path, 'resources', 'models_zoo.yaml')
    model_config = yaml.load(open(model_config, 'r'),
                             Loader=yaml.FullLoader)
    models = list(model_config.keys())
    return models