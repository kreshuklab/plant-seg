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


def test_update_halo(segmentation_tab):
    w_unet = segmentation_tab.prediction_widgets.widget_unet_prediction
    w_unet.advanced.value = True
    w_unet.mode.value = UNetPredictionMode.PLANTSEG
    w_unet.model_name.value = "generic_confocal_3D_unet"
    segmentation_tab.prediction_widgets.update_halo()
    assert w_unet.patch_size.value == (128, 128, 128)
    assert w_unet.patch_size[0].enabled
    assert w_unet.patch_halo.value == (44, 44, 44)
    assert w_unet.patch_halo[0].enabled

    w_unet.model_name.value = "confocal_2D_unet_ovules_ds2x"
    segmentation_tab.prediction_widgets.update_halo()
    assert w_unet.patch_size.value == (1, 128, 128)
    assert not w_unet.patch_size[0].enabled
    assert w_unet.patch_halo.value == (0, 44, 44)
    assert not w_unet.patch_halo[0].enabled


def test_update_halo_biio(segmentation_tab, mocker):
    mocked_log = mocker.patch(
        target="plantseg.viewer_napari.widgets.prediction.log",
        autospec=True,
    )
    w_unet = segmentation_tab.prediction_widgets.widget_unet_prediction
    w_unet.advanced.value = True
    w_unet.mode.value = UNetPredictionMode.BIOIMAGEIO

    segmentation_tab.prediction_widgets.update_halo()
    mocked_log.assert_called_with(
        "Automatic halo not implemented for BioImage.IO models yet "
        "because they are handled by BioImage.IO Core.",
        thread="BioImage.IO Core prediction",
        level="DEBUG",
    )


def test_on_model_name_change_model_no_model(segmentation_tab, mocker):
    mocked_show = mocker.patch.object(
        segmentation_tab.prediction_widgets, "show_add_custom_model"
    )
    segmentation_tab.prediction_widgets.widget_unet_prediction.model_name.value = None
    mocked_show.assert_not_called()


def test_on_model_name_change_custom_model(segmentation_tab, mocker):
    mocked_show = mocker.patch.object(
        segmentation_tab.prediction_widgets, "show_add_custom_model"
    )
    segmentation_tab.prediction_widgets.widget_unet_prediction.model_name.value = (
        segmentation_tab.prediction_widgets.ADD_MODEL
    )
    mocked_show.assert_called_once()


def test_on_advanced_changed(segmentation_tab, mocker):
    mocked_halo = mocker.patch.object(
        segmentation_tab.prediction_widgets, "update_halo"
    )
    mocked_show = mocker.patch.object(
        segmentation_tab.prediction_widgets.advanced_unet_prediction_widgets[0], "show"
    )
    mocked_hide = mocker.patch.object(
        segmentation_tab.prediction_widgets.advanced_unet_prediction_widgets[0], "hide"
    )
    segmentation_tab.prediction_widgets.widget_unet_prediction.advanced.value = True
    mocked_halo.assert_called_once()
    mocked_show.assert_called_once()
    mocked_hide.assert_not_called()

    mocked_halo.reset_mock()
    mocked_show.reset_mock()
    mocked_hide.reset_mock()

    segmentation_tab.prediction_widgets.widget_unet_prediction.advanced.value = False

    mocked_halo.assert_not_called()
    mocked_show.assert_not_called()
    mocked_hide.assert_called_once()


def test_on_widget_unet_mode_change(segmentation_tab, mocker):
    segmentation_tab.prediction_widgets.widget_unet_prediction.advanced.value = True
    mocked_halo = mocker.patch.object(
        segmentation_tab.prediction_widgets, "update_halo"
    )
    segmentation_tab.prediction_widgets.widget_unet_prediction.mode.value = (
        UNetPredictionMode.BIOIMAGEIO
    )
    mocked_halo.assert_called_once()

    mocked_halo.reset_mock()
    segmentation_tab.prediction_widgets.widget_unet_prediction.mode.value = (
        UNetPredictionMode.PLANTSEG
    )
    mocked_halo.assert_called_once()


def test_on_widget_unet_prediction_plantseg_filter_change(segmentation_tab, mocker):
    mocked_plantseg = mocker.patch(
        "plantseg.viewer_napari.widgets.prediction.model_zoo.get_bioimageio_zoo_plantseg_model_names"
    )
    mocked_other = mocker.patch(
        "plantseg.viewer_napari.widgets.prediction.model_zoo.get_bioimageio_zoo_other_model_names"
    )
    segmentation_tab.prediction_widgets.widget_unet_prediction.plantseg_filter.value = (
        False
    )
    mocked_plantseg.assert_called_once()
    mocked_other.assert_called_once()

    mocked_plantseg.reset_mock()
    mocked_other.reset_mock()

    segmentation_tab.prediction_widgets.widget_unet_prediction.plantseg_filter.value = (
        True
    )
    mocked_plantseg.assert_called_once()
    mocked_other.assert_not_called()


def test_on_any_metadata_changed(segmentation_tab):
    segmentation_tab.prediction_widgets.model_filters[0].value = "2D"
    segmentation_tab.prediction_widgets.model_filters[0].value = "3D"
