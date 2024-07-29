from plantseg.functionals.predictions import unet_predictions
from plantseg.tasks import task_tracker
from plantseg.plantseg_image import PlantSegImage, SemanticType
from pathlib import Path


@task_tracker
def unet_predictions_task(
    image: PlantSegImage,
    model_name: str | None,
    model_id: str | None,
    suffix: str = "_predictions",
    patch: tuple[int, int, int] = (80, 160, 160),
    single_batch_mode: bool = True,
    device: str = "cuda",
    model_update: bool = False,
    disable_tqdm: bool = False,
    handle_multichannel: bool = False,
    config_path: Path | None = None,
    model_weights_path: Path | None = None,
) -> PlantSegImage:
    """
    Apply a trained U-Net model to a PlantSegImage object.

    Args:
        image (PlantSegImage): input image
        model_path (str): path to the trained U-Net model
        device (str): device to use for prediction (default: "cpu")

    """
    data = image.data
    pmap = unet_predictions(
        raw=data,
        model_name=model_name,
        model_id=model_id,
        patch=patch,
        single_batch_mode=single_batch_mode,
        device=device,
        model_update=model_update,
        disable_tqdm=disable_tqdm,
        handle_multichannel=handle_multichannel,
        config_path=config_path,
        model_weights_path=model_weights_path,
    )

    new_image = image.derive_new(pmap, name=f"{image.name}_{suffix}")
    new_image.semantic_type = SemanticType.PREDICTION
    return new_image
