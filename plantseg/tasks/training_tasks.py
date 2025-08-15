import logging
from pathlib import Path
from typing import Literal, Optional

from plantseg import PATH_PLANTSEG_MODELS
from plantseg.functionals.training.train import unet_training
from plantseg.tasks.workflow_handler import task_tracker
from plantseg.viewer_napari import log


@task_tracker
def unet_training_task(
    dataset_dir: str | Path,
    model_name: str,
    in_channels: int,
    out_channels: int,
    feature_maps: int | list[int] | tuple[int, ...],
    patch_size: tuple[int, int, int],
    max_num_iters: int,
    dimensionality: Literal["2D", "3D"],
    sparse: bool,
    device: str,
    _tracker: Optional["PBar_Tracker"] = None,
):
    unet_training(
        dataset_dir=dataset_dir,
        model_name=model_name,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_maps=feature_maps,
        patch_size=patch_size,
        max_num_iters=max_num_iters,
        dimensionality=dimensionality,
        sparse=sparse,
        device=device,
    )

    checkpoint_dir = PATH_PLANTSEG_MODELS / model_name
    log(f"Finished training, saved model to {checkpoint_dir}", thread="train_gui")
