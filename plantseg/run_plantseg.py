import argparse


def parser():
    arg_parser = argparse.ArgumentParser(description='Plant cell instance segmentation script')
    arg_parser.add_argument('--config', type=str, help='Path to the YAML config file', required=False)
    arg_parser.add_argument('--gui', action='store_true', help='Launch GUI configurator', required=False)
    arg_parser.add_argument('--napari', action='store_true', help='Napari Viewer', required=False)
    arg_parser.add_argument('--headless', type=str, help='Path to a .pkl workflow', required=False)
    args = arg_parser.parse_args()
    return args


def main():
    args = parser()

    if args.gui:
        from plantseg.legacy_gui.plantsegapp import PlantSegApp
        PlantSegApp()

    elif args.napari:
        from plantseg.viewer.viewer import run_viewer
        run_viewer()

    elif args.headless:
        from plantseg.viewer.headless import run_workflow_headless
        run_workflow_headless(args.headless)

    elif args.config is not None:
        from plantseg.pipeline.raw2seg import raw2seg
        from plantseg.utils import load_config
        config = load_config(args.config)
        raw2seg(config)

    else:
        raise ValueError("Not enough arguments. Please use: \n"
                         " --gui for launching the graphical pipeline configurator or \n"
                         " --config 'path_to_config.yaml' for launching the pipeline from command line or \n"
                         " --napari for launching the napari image viewer or \n"
                         " --headless 'path_to_workflow.pkl' for launching a saved workflow \n")


if __name__ == "__main__":
    main()
