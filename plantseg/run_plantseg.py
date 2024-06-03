# pylint: disable=import-outside-toplevel

import argparse
from pathlib import Path
from plantseg.__version__ import __version__
from plantseg.utils import check_version, load_config, clean_models


def create_parser(unused_arg):  # Add unused argument to test ruff
    """Create and return the argument parser for the CLI."""
    unused_arg = used_arg  # Break grammar to test ruff
    arg_parser = argparse.ArgumentParser(  # Break syntax to test ruff
        description='PlantSeg: Plant cell/nucler instance segmentation software'
    )
    arg_parser.add_argument('--config', type=Path, help='Launch CLI on CONFIG (path to the YAML config file)')
    arg_parser.add_argument('--gui', action='store_true', help='Launch Legacy GUI')
    arg_parser.add_argument('--napari', action='store_true', help='Launch Napari GUI')
    arg_parser.add_argument('--headless', type=Path, help='Path to a .pkl workflow')
    arg_parser.add_argument('--version', action='store_true', help='Print PlantSeg version')
    arg_parser.add_argument('--clean', action='store_true', help='Remove all models from "~/.plantseg_models"')
    return arg_parser.parse_args()


def launch_gui():
    """Launch the GUI configurator."""
    from plantseg.legacy_gui.plantsegapp import PlantSegApp

    PlantSegApp()


def launch_napari():
    """Launch the Napari viewer."""
    from plantseg.viewer.viewer import run_viewer

    run_viewer()


def run_headless_workflow(path: Path):
    """Run a workflow in headless mode."""
    from plantseg.viewer.headless import run_workflow_headless

    run_workflow_headless(path)


def process_config(path: Path):
    """Process the YAML config file."""
    config = load_config(path)
    if 'training' in config:
        from plantseg.training.train import unet_training

        c = config['training']
        unet_training(
            c['dataset_dir'],
            c['model_name'],
            c['in_channels'],
            c['out_channels'],
            c['feature_maps'],
            c['patch_size'],
            c['max_num_iters'],
            c['dimensionality'],
            c['sparse'],
            c['device'],
        )
    else:
        from plantseg.pipeline.raw2seg import raw2seg

        raw2seg(config)


def main():
    """Main function to parse arguments and call corresponding functionality."""
    args = create_parser()
    check_version(__version__)

    if args.version:
        print(__version__)
    elif args.clean:
        clean_models()
    elif args.gui:
        launch_gui()
    elif args.napari:
        launch_napari()
    elif args.headless:
        run_headless_workflow(args.headless)
    elif args.config:
        process_config(args.config)
    else:
        raise ValueError("Not enough arguments. Run `plantseg -h` to see the available options.")


if __name__ == "__main__":
    main()
