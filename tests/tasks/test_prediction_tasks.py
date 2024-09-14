import numpy as np
import pytest

from plantseg.core import ImageProperties, PlantSegImage, VoxelSize
from plantseg.core.image import ImageLayout, SemanticType
from plantseg.tasks.prediction_tasks import unet_prediction_task


@pytest.mark.parametrize(
    "shape, layout, model_name",
    [
        ((32, 64, 64), ImageLayout.ZYX, 'generic_confocal_3D_unet'),
        ((64, 64), ImageLayout.YX, 'confocal_2D_unet_ovules_ds2x'),
    ],
)
def test_unet_prediction(shape, layout, model_name):
    mock_data = np.random.rand(*shape).astype('float32')

    property = ImageProperties(
        name='test',
        voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit='um'),
        semantic_type=SemanticType.RAW,
        image_layout=layout,
        original_voxel_size=VoxelSize(voxels_size=(1.0, 1.0, 1.0), unit='um'),
    )
    image = PlantSegImage(data=mock_data, properties=property)

    result = unet_prediction_task(image=image, model_name=model_name, model_id=None, device='cpu')

    assert len(result) == 1
    result = result[0]

    assert result.semantic_type == SemanticType.PREDICTION
    assert result.image_layout == property.image_layout
    assert result.voxel_size == property.voxel_size
    assert result.shape == mock_data.shape
