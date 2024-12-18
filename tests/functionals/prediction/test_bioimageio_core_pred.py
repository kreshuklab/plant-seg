from pathlib import Path

import pooch
import pytest
import skimage.transform as skt

from plantseg.functionals.prediction.prediction import biio_prediction
from plantseg.io.io import smart_load

CELLPOSE_TEST_IMAGE_RGB_3D = 'http://www.cellpose.org/static/data/rgb_3D.tif'
path_rgb_3d_75x2x75x75 = Path(pooch.retrieve(CELLPOSE_TEST_IMAGE_RGB_3D, known_hash=None))
raw_zcyx_75x2x75x75 = smart_load(path_rgb_3d_75x2x75x75)
raw_zcyx_96x2x96x96 = skt.resize(raw_zcyx_75x2x75x75, (96, 2, 96, 96), order=1)
raw_cell_3d_100x128x128 = skt.resize(raw_zcyx_75x2x75x75[:, 1], (100, 128, 128), order=1)
raw_cell_2d_96x96 = raw_cell_3d_100x128x128[48]


@pytest.mark.parametrize(
    "raw, input_layout, model_id",
    (
        (raw_zcyx_96x2x96x96, 'ZCYX', 'philosophical-panda'),
        (raw_cell_3d_100x128x128, 'ZYX', 'emotional-cricket'),
        (raw_cell_2d_96x96, 'YX', 'pioneering-rhino'),
    ),
)
def test_biio_prediction(raw, input_layout, model_id):
    named_pmaps = biio_prediction(raw, input_layout, model_id)
    for key, pmap in named_pmaps.items():
        assert pmap is not None, f"Prediction map for {key} is None"
        assert pmap.ndim == 4, f"Prediction map for {key} has {pmap.ndim} dimensions"
