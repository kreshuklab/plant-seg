from PIL import Image
from PIL.ImageOps import grayscale
import numpy as np
from pathlib import Path


PIL_EXTENSIONS = [".png", ".jpg", ".jpeg"]


def load_pil(path: Path) -> np.ndarray:
    """
    Load a tiff file using PIL

    Args:
        path (Path): path to an image file to load (png, jpg, jpeg).
        info_only (bool, optional): if true will return a tuple with the default infos.
            true infos can not be extracted from a PIL image. Defaults to False.
    Returns:
        Union[tuple, tuple[np.ndarray, tuple]]: loaded data as numpy array and infos
    """

    image = Image.open(path)
    if image.mode != "L":
        image = grayscale(image)

    image = np.array(image)

    return image
