import argparse

import yaml

from plantseg.legacy_gui.plantsegapp import PlantSegApp
from plantseg.pipeline.raw2seg import raw2seg
from plantseg.viewer.headless import run_workflow_headless
from plantseg.viewer.viewer import run_viewer


def parser():
    parser = argparse.ArgumentParser(description='Plant cell instance segmentation script')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=False)
    parser.add_argument('--gui', action='store_true', help='Launch GUI configurator', required=False)
    parser.add_argument('--napari', action='store_true', help='Napari Viewer', required=False)
    parser.add_argument('--headless', type=str, help='Run headless workflow', required=False)
    args = parser.parse_args()
    return args


def load_config(args):
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    return config


def main():
    args = parser()

    if args.gui:
        PlantSegApp()

    elif args.napari:
        run_viewer()

    elif args.headless:
        run_workflow_headless(args.headless)

    elif args.config is not None:
        config = load_config(args)
        raw2seg(config)

    else:
        raise ValueError("Not enough arguments. Please use: \n"
                         " --legacy_gui for launching the graphical pipeline configurator or \n"
                         " --config 'path_to_config.yaml' for launching the pipeline from command line")


if __name__ == "__main__":
    main()
