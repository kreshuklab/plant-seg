import napari
import numpy as np
import pytest

from plantseg._viewer.widget.dataprocessing import RescaleModes, WidgetName, widget_rescaling
from plantseg._viewer.widget.utils import create_layer_name


@pytest.fixture
def sample_image():
    return {
        "data": np.random.random((10, 100, 100)),
        "name": "sample_image",
        "scale": (1.0, 1.0, 1.0),
    }


@pytest.fixture
def sample_label():
    return {
        "data": np.random.randint(0, 2, (5, 100, 100)),
        "name": "sample_label",
        "scale": (2.0, 1.0, 1.0),
    }


class TestWidgetRescaling:
    def test_rescaling_from_factor(self, make_napari_viewer_proxy, sample_image):
        viewer = make_napari_viewer_proxy()
        viewer.add_image(**sample_image)

        factor = 0.5
        widget_rescaling(
            viewer=viewer,
            image=viewer.layers[sample_image['name']],
            mode=RescaleModes.FROM_FACTOR,
            rescaling_factor=(factor, factor, factor),
            update_other_widgets=False,
        )
        napari.run()

        new_layer_name = create_layer_name(sample_image['name'], WidgetName.RESCALING.layer_suffix)
        new_layer = viewer.layers[new_layer_name]
        old_layer = viewer.layers[sample_image['name']]

        np.testing.assert_allclose(new_layer.data.shape, np.multiply(old_layer.data.shape, factor), rtol=1e-5)
        np.testing.assert_allclose(np.multiply(new_layer.scale, factor), old_layer.scale, rtol=1e-5)

    def test_rescaling_to_layer_voxel_size(self, make_napari_viewer_proxy, sample_image, sample_label):
        viewer = make_napari_viewer_proxy()
        viewer.add_image(**sample_image)
        viewer.add_image(**sample_label)

        widget_rescaling(
            viewer=viewer,
            image=viewer.layers[sample_image['name']],
            mode=RescaleModes.TO_LAYER_VOXEL_SIZE,
            reference_layer=viewer.layers[sample_label['name']],
            update_other_widgets=False,
        )
        napari.run()
        new_layer_name = create_layer_name(sample_image['name'], WidgetName.RESCALING.layer_suffix)

        new_layer = viewer.layers[new_layer_name]
        reference_layer = viewer.layers[sample_label['name']]

        np.testing.assert_allclose(new_layer.data.shape, reference_layer.data.shape, rtol=1e-5)
        np.testing.assert_allclose(new_layer.scale, reference_layer.scale, rtol=1e-5)

    def test_rescaling_to_model_voxel_size(self, make_napari_viewer_proxy, sample_image):
        viewer = make_napari_viewer_proxy()
        viewer.add_image(**sample_image)

        widget_rescaling(
            viewer=viewer,
            image=viewer.layers[sample_image['name']],
            mode=RescaleModes.TO_MODEL_VOXEL_SIZE,
            reference_model='PlantSeg_3Dnuc_platinum',  # voxel size: (0.2837, 0.1268, 0.1268)
            update_other_widgets=False,
        )
        napari.run()

        new_layer_name = create_layer_name(sample_image['name'], WidgetName.RESCALING.layer_suffix)
        new_layer = viewer.layers[new_layer_name]
        old_layer = viewer.layers[sample_image['name']]

        expected_scale = (0.2837, 0.1268, 0.1268)

        np.testing.assert_allclose(new_layer.scale, expected_scale, rtol=1e-5)

        new_shape = np.array(new_layer.data.shape)
        new_scale = np.array(new_layer.scale)
        old_shape = np.array(old_layer.data.shape)
        old_scale = np.array(old_layer.scale)

        np.testing.assert_allclose(new_shape * new_scale, old_shape * old_scale, atol=0.1)

    def test_rescaling_to_voxel_size(self, make_napari_viewer_proxy, sample_image):
        viewer = make_napari_viewer_proxy()
        viewer.add_image(**sample_image)

        expected_scale = (0.5, 0.5, 0.5)  # target voxel size
        widget_rescaling(
            viewer=viewer,
            image=viewer.layers[sample_image['name']],
            mode=RescaleModes.TO_VOXEL_SIZE,
            out_voxel_size=expected_scale,
            update_other_widgets=False,
        )
        napari.run()

        new_layer_name = create_layer_name(sample_image['name'], WidgetName.RESCALING.layer_suffix)
        new_layer = viewer.layers[new_layer_name]
        old_layer = viewer.layers[sample_image['name']]

        scaling_factor = np.array(old_layer.scale) / np.array(expected_scale)
        expected_shape = np.round(np.array(old_layer.data.shape) * scaling_factor).astype(int)

        np.testing.assert_allclose(new_layer.data.shape, expected_shape, rtol=1e-5)
        np.testing.assert_allclose(new_layer.scale, expected_scale, rtol=1e-5)

    def test_rescaling_to_layer_shape(self, make_napari_viewer_proxy, sample_image, sample_label):
        viewer = make_napari_viewer_proxy()
        viewer.add_image(**sample_image)
        viewer.add_image(**sample_label)

        widget_rescaling(
            viewer=viewer,
            image=viewer.layers[sample_image['name']],
            mode=RescaleModes.TO_LAYER_SHAPE,
            reference_layer=viewer.layers[sample_label['name']],
            update_other_widgets=False,
        )
        napari.run()

        new_layer_name = create_layer_name(sample_image['name'], WidgetName.RESCALING.layer_suffix)
        new_layer = viewer.layers[new_layer_name]
        reference_layer = viewer.layers[sample_label['name']]

        np.testing.assert_allclose(new_layer.data.shape, reference_layer.data.shape, rtol=1e-5)

        # In our special case in this test, yes, the scale should be the same
        np.testing.assert_allclose(new_layer.scale, reference_layer.scale, rtol=1e-5)

    def test_rescaling_set_shape(self, make_napari_viewer_proxy, sample_image):
        viewer = make_napari_viewer_proxy()
        viewer.add_image(**sample_image)

        target_shape = (20, 50, 50)
        widget_rescaling(
            viewer=viewer,
            image=viewer.layers[sample_image['name']],
            mode=RescaleModes.TO_SHAPE,
            reference_shape=target_shape,
            update_other_widgets=False,
        )
        napari.run()

        new_layer_name = create_layer_name(sample_image['name'], WidgetName.RESCALING.layer_suffix)
        new_layer = viewer.layers[new_layer_name]
        old_layer = viewer.layers[sample_image['name']]

        np.testing.assert_allclose(new_layer.data.shape, target_shape, rtol=1e-5)

        scaling_factor = np.array(target_shape) / np.array(old_layer.data.shape)
        expected_scale = np.array(old_layer.scale) / scaling_factor

        np.testing.assert_allclose(new_layer.scale, expected_scale, rtol=1e-5)

    def test_rescaling_set_voxel_size(self, make_napari_viewer_proxy, sample_image):
        viewer = make_napari_viewer_proxy()
        viewer.add_image(**sample_image)

        target_voxel_size = (2.0, 2.0, 2.0)
        widget_rescaling(
            viewer=viewer,
            image=viewer.layers[sample_image['name']],
            mode=RescaleModes.SET_VOXEL_SIZE,
            out_voxel_size=target_voxel_size,
            update_other_widgets=False,
        )

        old_layer = viewer.layers[sample_image['name']]
        np.testing.assert_allclose(old_layer.scale, target_voxel_size, rtol=1e-5)
