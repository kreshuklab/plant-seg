import pytest

from plantseg.tasks.prediction_tasks import biio_prediction_task, unet_prediction_task
from plantseg.viewer_napari.widgets.prediction import UNetPredictionMode
from plantseg.viewer_napari.widgets.segmentation import Segmentation_Tab


@pytest.fixture
def segmentation_tab():
    return Segmentation_Tab()


def test_widget_unet_prediction_no_layer(segmentation_tab, mocker):
    mocked_scheduler = mocker.patch(
        target="plantseg.viewer_napari.widgets.prediction.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="plantseg.viewer_napari.widgets.prediction.log",
        autospec=True,
    )

    segmentation_tab.widget_layer_select.layer.choices = (None,)
    segmentation_tab.widget_layer_select.layer.value = None

    segmentation_tab.prediction_widgets.widget_unet_prediction()
    mocked_scheduler.assert_not_called()
    mocked_log.assert_called_with(
        "Please load an image first!", thread="Prediction", level="WARNING"
    )


def test_widget_unet_prediction_no_model(segmentation_tab, mocker, napari_raw):
    mocked_scheduler = mocker.patch(
        target="plantseg.viewer_napari.widgets.prediction.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="plantseg.viewer_napari.widgets.prediction.log",
        autospec=True,
    )

    segmentation_tab.widget_layer_select.layer.choices = (napari_raw,)
    segmentation_tab.widget_layer_select.layer.value = napari_raw

    segmentation_tab.prediction_widgets.widget_unet_prediction(
        mode=UNetPredictionMode.PLANTSEG, model_name=None
    )
    mocked_scheduler.assert_not_called()
    mocked_log.assert_called_with(
        "Choose a model first!",
        thread="Prediction",
        level="WARNING",
    )


def test_widget_unet_prediction_plantseg(segmentation_tab, mocker, napari_raw):
    mocked_scheduler = mocker.patch(
        target="plantseg.viewer_napari.widgets.prediction.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="plantseg.viewer_napari.widgets.prediction.log",
        autospec=True,
    )

    segmentation_tab.widget_layer_select.layer.choices = (napari_raw,)
    segmentation_tab.widget_layer_select.layer.value = napari_raw

    segmentation_tab.prediction_widgets.widget_unet_prediction(
        mode=UNetPredictionMode.PLANTSEG, model_name="smth"
    )
    assert unet_prediction_task == mocked_scheduler.call_args[0][0]
    mocked_log.assert_not_called()


def test_widget_unet_prediction_bioimageio(segmentation_tab, mocker, napari_raw):
    mocked_scheduler = mocker.patch(
        target="plantseg.viewer_napari.widgets.prediction.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="plantseg.viewer_napari.widgets.prediction.log",
        autospec=True,
    )

    segmentation_tab.widget_layer_select.layer.choices = (napari_raw,)
    segmentation_tab.widget_layer_select.layer.value = napari_raw

    segmentation_tab.prediction_widgets.widget_unet_prediction(
        mode=UNetPredictionMode.BIOIMAGEIO, model_id="smth"
    )
    assert biio_prediction_task == mocked_scheduler.call_args[0][0]
    mocked_log.assert_not_called()
