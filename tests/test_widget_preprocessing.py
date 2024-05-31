import numpy as np
import napari
from plantseg.viewer.widget.dataprocessing import widget_rescaling, RescaleModes, RescaleType


class TestWidgetRescaling:

    def test_rescaling_from_factor(self, make_napari_viewer_proxy, sample_image):
        viewer = make_napari_viewer_proxy()
        viewer.add_image(np.random.rand(10, 100, 100).astype(np.float32), name="sample_image", scale=(1.0, 1.0, 1.0))
        widget_rescaling(
            viewer=viewer,
            image=sample_image,
            mode=RescaleModes.FROM_FACTOR,
            rescaling_factor=(0.5, 0.5, 0.5),
        )
        napari.run()
        print(viewer.layers)
        assert viewer.layers["sample_image_Rescaled"].data.shape == (5, 50, 50)
