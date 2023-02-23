from PIL import Image
from PIL.ImageOps import grayscale
import numpy as np
from typing import Union

PIL_EXTENSIONS = ['.png', '.jpg', '.jpeg']


def load_pill(path: str, info_only=False) -> Union[tuple, tuple[np.array, tuple]]:
    """
    Load a tiff file using PIL
    """

    image = Image.open(path)
    if image.mode != 'L':
        image = grayscale(image)

    image = np.array(image)

    # create infos
    infos = ([1., 1., 1.], image.shape, None, 'um')

    if info_only:
        return infos

    return image, infos
