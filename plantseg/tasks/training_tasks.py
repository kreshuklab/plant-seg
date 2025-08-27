import tempfile
from pathlib import Path
from typing import Literal, Optional

from napari.layers import Image, Labels

from plantseg import PATH_PLANTSEG_MODELS, logger
from plantseg.core.image import PlantSegImage, save_image
from plantseg.functionals.training.train import unet_training
from plantseg.tasks.workflow_handler import task_tracker
from plantseg.viewer_napari import log


@task_tracker
def unet_training_task(
    dataset_dir: Optional[Path],
    image: Optional[Image],
    segmentation: Optional[Labels],
    model_name: str,
    in_channels: int,
    out_channels: int,
    feature_maps: int | list[int] | tuple[int, ...],
    patch_size: tuple[int, int, int],
    max_num_iters: int,
    dimensionality: Literal["2D", "3D"],
    sparse: bool,
    device: str,
    modality: str = "",
    output_type: str = "",
    description: str = "",
    resolution: tuple[float, float, float] = (1.0, 1.0, 1.0),
    pre_trained: Path | None = None,
    widgets_to_reset: Optional[list] = None,
    _tracker: Optional["PBar_Tracker"] = None,
):
    if dataset_dir is None and (image is None or segmentation is None):
        raise ValueError("dataset_dir or (image and segmentation) must not be None!")
    if dataset_dir is not None and (image is not None or segmentation is not None):
        raise ValueError("dataset_dir or (image and segmentation) must be None!")

    # make training dataset on the fly
    tmp_dir = None
    if image is not None and segmentation is not None:
        tmp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        out_path = Path(tmp_dir.name)
        train = out_path / "train"
        train.mkdir()
        val = out_path / "val"
        val.mkdir()
        save_image(
            image=PlantSegImage.from_napari_layer(image),
            export_directory=train,
            name_pattern="to_train",
            key="raw",
            scale_to_origin=False,
            export_format="h5",
        )
        save_image(
            image=PlantSegImage.from_napari_layer(segmentation),
            export_directory=train,
            name_pattern="to_train",
            key="label",
            scale_to_origin=False,
            export_format="h5",
        )
        dataset_dir = out_path

    assert dataset_dir is not None, (
        "Logic error in unet_training_task"
    )  # make pyright happy
    if not all((dataset_dir / d).exists() for d in ["train", "val"]):
        log(
            "Dataset dir must contain a train and a val directory,\n"
            "each containing h5 files!",
            thread="train_gui_task",
        )
        return

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
        modality=modality,
        output_type=output_type,
        description=description,
        resolution=resolution,
        pre_trained=pre_trained,
    )

    checkpoint_dir = PATH_PLANTSEG_MODELS / model_name
    log(f"Finished training, saved model to {checkpoint_dir}", thread="train_gui_task")
    if widgets_to_reset:
        for widget in widgets_to_reset:
            logger.debug(f"Updating: {widget}")
            widget.reset_choices()

    if tmp_dir:
        tmp_dir.cleanup()
