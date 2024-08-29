from pathlib import Path

from plantseg.functionals.predictions import unet_predictions
from plantseg.plantseg_image import ImageLayout, PlantSegImage, SemanticType
from plantseg.tasks import task_tracker


@task_tracker
def unet_predictions_task(
    image: PlantSegImage,
    model_name: str | None,
    model_id: str | None,
    suffix: str = "_predictions",
    patch: tuple[int, int, int] = (80, 160, 160),
    patch_halo: tuple[int, int, int] | None = None,
    single_batch_mode: bool = True,
    device: str = "cuda",
    model_update: bool = False,
    disable_tqdm: bool = False,
    config_path: Path | None = None,
    model_weights_path: Path | None = None,
) -> list[PlantSegImage]:
    """
    Apply a trained U-Net model to a PlantSegImage object.

    Args:
        image (PlantSegImage): input image object
        model_name (str): the name of the model to use
        model_id (str): the ID of the model to use
        suffix (str): suffix to append to the new image name
        patch (tuple[int, int, int]): patch size for prediction
        single_batch_mode (bool): whether to use a single batch for prediction
        device (str): the computation device ('cpu', 'cuda', etc.)
        model_update (bool): whether to update the model to the latest version
    """
    data = image.get_data()
    pmaps = unet_predictions(
        raw=data,
        model_name=model_name,
        model_id=model_id,
        patch=patch,
        patch_halo=patch_halo,
        single_batch_mode=single_batch_mode,
        device=device,
        model_update=model_update,
        disable_tqdm=disable_tqdm,
        handle_multichannel=True,  # always receive a (C, Z, Y, X) prediction
        config_path=config_path,
        model_weights_path=model_weights_path,
    )
    assert pmaps.ndim == 4, f"Expected 4D prediction, got {pmaps.ndim}D"

    new_images = []
    image_layout = ImageLayout.ZYX if pmaps.shape[1] > 1 else ImageLayout.YX
    for i, pmap in enumerate(pmaps):
        new_images.append(
            image.derive_new(
                pmap.squeeze(),  # pmap is a (Z, Y, X) prediction, Z >= 1
                name=f"{image.name}_{suffix}_{i}",
                semantic_type=SemanticType.PREDICTION,
                image_layout=image_layout,
            )
        )

    return new_images
