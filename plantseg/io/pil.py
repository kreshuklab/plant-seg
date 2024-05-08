from PIL import Image
from PIL.ImageOps import grayscale
import numpy as np
from typing import Union

PIL_EXTENSIONS = ['.png', '.jpg', '.jpeg']


def load_pill(path: str, info_only=False) -> Union[tuple, tuple[np.ndarray, tuple]]:
    """
    Load a tiff file using PIL

    Args:
        path (str): Path to the image file [png, jpg, jpeg]
        info_only (bool, optional): if true will return a tuple with the default infos.
            true infos can not be extracted from a PIL image. Defaults to False.
    Returns:
        Union[tuple, tuple[np.ndarray, tuple]]: loaded data as numpy array and infos
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
