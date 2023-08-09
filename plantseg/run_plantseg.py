import argparse


def parser():
    arg_parser = argparse.ArgumentParser(description='Plant cell instance segmentation script')
    arg_parser.add_argument('--config', type=str, help='Path to the YAML config file', required=False)
    arg_parser.add_argument('--gui', action='store_true', help='Launch GUI configurator', required=False)
    arg_parser.add_argument('--napari', action='store_true', help='Napari Viewer', required=False)
    arg_parser.add_argument('--training', action='store_true', help='Train a plantseg model', required=False)
    arg_parser.add_argument('--headless', type=str, help='Path to a .pkl workflow', required=False)
    arg_parser.add_argument('--version', action='store_true', help='PlantSeg version', required=False)
    arg_parser.add_argument('--clean', action='store_true',
                            help='Remove all models from "~/.plantseg_models"', required=False)
    args = arg_parser.parse_args()
    return args


def main():
    from plantseg.utils import check_version
    check_version()
    args = parser()

    if args.gui:
        from plantseg.legacy_gui.plantsegapp import PlantSegApp
        PlantSegApp()

    elif args.napari:
        from plantseg.viewer.viewer import run_viewer
        run_viewer()

    elif args.training:
        from plantseg.viewer.training import run_training_headless
        run_training_headless()

    elif args.headless:
        from plantseg.viewer.headless import run_workflow_headless
        run_workflow_headless(args.headless)

    elif args.config is not None:
        from plantseg.pipeline.raw2seg import raw2seg
        from plantseg.utils import load_config
        config = load_config(args.config)
        raw2seg(config)

    elif args.version:
        from plantseg.__version__ import __version__
        print(__version__)

    elif args.clean:
        from plantseg.utils import clean_models
        clean_models()

    else:
        raise ValueError("Not enough arguments. Please use: \n"
                         " --napari for launching the napari image viewer or \n"
                         " --training for launching the training configurator or \n"
                         " --headless 'path_to_workflow.pkl' for launching a saved workflow or \n"
                         " --gui for launching the graphical pipeline configurator or \n"
                         " --config 'path_to_config.yaml' for launching the pipeline from command line or \n"
                         " --version for printing the PlantSeg version")


if __name__ == "__main__":
    main()
