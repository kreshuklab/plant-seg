import argparse
import yaml
from plantseg.pipeline.raw2seg import raw2seg
from plantseg.gui.plantsegapp import PlantSegApp
from plantseg.rest.server import run_server


def parser():
    parser = argparse.ArgumentParser(description='Plant cell instance segmentation script')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=False)
    parser.add_argument('--gui', action='store_true', help='Launch GUI configurator', required=False)
    parser.add_argument('--server', action='store_true', help='Run PlantSeg REST API server', required=False)
    parser.add_argument('--datadir', type=str, help='Path to the directory where the data are stored', required=False)
    args = parser.parse_args()
    return args


def load_config(args):
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    return config


def main():
    args = parser()

    if args.gui:
        assert not args.server, 'PlantSeg cannot be started in the GUI and server mode at the same time'
        PlantSegApp()
    elif args.config is not None:
        config = load_config(args)
        raw2seg(config)
    elif args.server:
        assert args.datadir is not None, 'datadir has to be provided in server mode'
        run_server(args.datadir)
    else:
        raise ValueError("Not enough arguments. Please use: \n"
                         " --gui for launching the graphical pipeline configurator or \n"
                         " --config 'path_to_config.yaml' for launching the pipeline from command line")


if __name__ == "__main__":
    main()
