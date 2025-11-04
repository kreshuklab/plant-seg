# pylint: disable=import-outside-toplevel

import argparse
from pathlib import Path
from typing import Optional

from plantseg import logger
from plantseg.__version__ import __version__
from plantseg.utils import check_version, clean_models, load_config


def create_parser():
    """Create and return the argument parser for the CLI."""
    arg_parser = argparse.ArgumentParser(
        description="PlantSeg: Plant cell/nucler instance segmentation software"
    )
    arg_parser.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Launch CLI from CONFIG (path to the YAML config file)",
    )
    arg_parser.add_argument(
        "--napari",
        "-n",
        action="store_true",
        help="Launch Napari GUI",
    )
    arg_parser.add_argument(
        "--train",
        type=Path,
        help="Launch training from CONFIG (path to the YAML config file)",
    )
    arg_parser.add_argument(
        "--version",
        "-v",
        action="store_true",
        help="Print PlantSeg version",
    )
    arg_parser.add_argument(
        "--clean",
        action="store_true",
        help='Remove all models from "~/.plantseg_models"',
    )
    arg_parser.add_argument(
        "--edit",
        "-e",
        type=Path,
        nargs="?",
        default=False,
        const=None,
        help="Lauch GUI to edit a workflow yaml file. Optionally provide a path.",
        metavar="yaml",
    )
    arg_parser.add_argument(
        "--loglevel",
        choices=["ERROR", "WARNING", "INFO", "DEBUG"],
        help="Set the level of the logger.",
    )
    return arg_parser.parse_args()


def launch_napari():
    """Launch the Napari viewer."""
    import rich.traceback

    from plantseg.viewer_napari.viewer import Plantseg_viewer

    rich.traceback.install(
        show_locals=True,
        suppress=[],
    )

    Plantseg_viewer().start_viewer()


def launch_workflow_headless(path: Path):
    """Run a workflow in headless mode."""
    from plantseg.headless.headless import run_headless_workflow

    run_headless_workflow(path=path)


def launch_training(path: Path):
    """Launch the training"""
    config = load_config(path)

    from plantseg.functionals.training.train import unet_training

    unet_training(*config)


def launch_editor(path: Optional[Path]):
    """Launch the workflow editor"""
    from plantseg.workflow_gui.editor import Workflow_gui

    Workflow_gui(path)


def main():
    """Main function to parse arguments and call corresponding functionality."""
    args = create_parser()
    check_version(__version__)

    if args.loglevel:
        logger.setLevel(args.loglevel)
        logger.info(f"Setting loglevel to {args.loglevel}.")

    if args.version:
        print(__version__)
    elif args.clean:
        clean_models()
    elif args.napari:
        launch_napari()
    elif args.config:
        launch_workflow_headless(args.config)
    elif args.train:
        launch_training(args.train)
    elif args.edit or args.edit is None:
        launch_editor(args.edit)
    else:
        raise ValueError(
            "Not enough arguments. Run `plantseg -h` to see the available options."
        )


if __name__ == "__main__":
    main()
