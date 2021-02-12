import yaml
from plantseg.pipeline import raw2seg_config_template
from functools import partial
from plantseg.pipeline import gui_logger
import os


def check_types(node, types_to_check=(str, )):
    if type(node) not in types_to_check:
        gui_logger.error("Type not a string")


def build_check(node):
    print(node)


def load_template():
    def _check(loader, node):
        if type(node) is dict:
            return build_check(node)
        elif type(node) is list:
            return [build_check(_node) for _node in node]
        else:
            raise RuntimeError("test")

    yaml.add_constructor('!check', _check)

    with open(raw2seg_config_template, 'r') as f:
        return yaml.full_load(f)


def config_validation(config):
    template = load_template()
    print(template)
    key_to_validate = ["path"]
    for key in key_to_validate:
        print(template[key](config[key]))
    #exit()
