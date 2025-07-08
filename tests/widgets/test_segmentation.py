import pytest

from plantseg.tasks.segmentation_tasks import (
    clustering_segmentation_task,
    lmc_segmentation_task,
)
from plantseg.viewer_napari.widgets.segmentation import Segmentation_Tab


@pytest.fixture
def segmentation_tab():
    return Segmentation_Tab()


def test_segmentation_initialization(segmentation_tab):
    containers = segmentation_tab.get_container()
    assert len(containers) == 12


def test_toggle_visibility_1(segmentation_tab, mocker):
    mocked_switch_2 = mocker.patch.object(segmentation_tab, "toggle_visibility_2")
    mocked_switch_3 = mocker.patch.object(segmentation_tab, "toggle_visibility_3")

    segmentation_tab.toggle_visibility_1(True)
    mocked_switch_2.assert_called_with(False)
    mocked_switch_3.assert_called_with(False)

    mocked_switch_2.reset_mock()
    mocked_switch_3.reset_mock()
    segmentation_tab.toggle_visibility_1(False)
    mocked_switch_2.assert_not_called()
    mocked_switch_3.assert_not_called()


def test_toggle_visibility_2(segmentation_tab, mocker):
    mocked_switch_1 = mocker.patch.object(segmentation_tab, "toggle_visibility_1")
    mocked_switch_3 = mocker.patch.object(segmentation_tab, "toggle_visibility_3")

    segmentation_tab.toggle_visibility_2(True)
    mocked_switch_1.assert_called_with(False)
    mocked_switch_3.assert_called_with(False)

    mocked_switch_1.reset_mock()
    mocked_switch_3.reset_mock()
    segmentation_tab.toggle_visibility_2(False)
    mocked_switch_1.assert_not_called()
    mocked_switch_3.assert_not_called()


def test_toggle_visibility_3(segmentation_tab, mocker):
    mocked_switch_1 = mocker.patch.object(segmentation_tab, "toggle_visibility_1")
    mocked_switch_2 = mocker.patch.object(segmentation_tab, "toggle_visibility_2")

    segmentation_tab.toggle_visibility_3(True)
    mocked_switch_1.assert_called_with(False)
    mocked_switch_2.assert_called_with(False)

    mocked_switch_1.reset_mock()
    mocked_switch_2.reset_mock()
    segmentation_tab.toggle_visibility_3(False)
    mocked_switch_1.assert_not_called()
    mocked_switch_2.assert_not_called()


def test_dt_watershed_no_prediction(segmentation_tab, mocker):
    mocked_scheduler = mocker.patch(
        target="plantseg.viewer_napari.widgets.segmentation.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="plantseg.viewer_napari.widgets.segmentation.log",
        autospec=True,
    )
    segmentation_tab.widget_layer_select.prediction.choices = (None,)
    segmentation_tab.widget_layer_select.prediction.value = None
    segmentation_tab.widget_dt_ws(stacked=False)
    mocked_scheduler.assert_not_called()
    mocked_log.assert_called_with(
        "Please run a boundary prediction first!",
        thread="Segmentation",
        level="WARNING",
    )


def test_dt_watershed(segmentation_tab, mocker, napari_prediction):
    mocked_scheduler = mocker.patch(
        target="plantseg.viewer_napari.widgets.segmentation.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="plantseg.viewer_napari.widgets.segmentation.log",
        autospec=True,
    )
    segmentation_tab.widget_layer_select.prediction.choices = (napari_prediction,)
    segmentation_tab.widget_layer_select.prediction.value = napari_prediction
    segmentation_tab.widget_dt_ws(stacked=False)
    mocked_scheduler.assert_called_once()
    mocked_log.assert_not_called()


def test_agglomeration_no_layer(segmentation_tab, mocker):
    mocked_scheduler = mocker.patch(
        target="plantseg.viewer_napari.widgets.segmentation.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="plantseg.viewer_napari.widgets.segmentation.log",
        autospec=True,
    )
    segmentation_tab.widget_layer_select.layer.choices = (None,)
    segmentation_tab.widget_layer_select.layer.value = None

    segmentation_tab.widget_agglomeration()
    mocked_scheduler.assert_not_called()
    mocked_log.assert_called_with(
        "Please load an input image first!",
        thread="Segmentation",
        level="WARNING",
    )


def test_agglomeration_no_superpixels(segmentation_tab, mocker, napari_prediction):
    mocked_scheduler = mocker.patch(
        target="plantseg.viewer_napari.widgets.segmentation.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="plantseg.viewer_napari.widgets.segmentation.log",
        autospec=True,
    )
    segmentation_tab.widget_layer_select.prediction.choices = (napari_prediction,)
    segmentation_tab.widget_layer_select.prediction.value = napari_prediction
    segmentation_tab.widget_layer_select.superpixels.choices = (None,)
    segmentation_tab.widget_layer_select.superpixels.value = None

    segmentation_tab.widget_agglomeration()
    mocked_scheduler.assert_not_called()
    mocked_log.assert_called_with(
        "Please run `Boundary to Superpixels` first!",
        thread="Segmentation",
        level="WARNING",
    )


def test_agglomeration(
    segmentation_tab, mocker, napari_prediction, napari_segmentation
):
    mocked_scheduler = mocker.patch(
        target="plantseg.viewer_napari.widgets.segmentation.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="plantseg.viewer_napari.widgets.segmentation.log",
        autospec=True,
    )
    segmentation_tab.widget_layer_select.prediction.choices = (napari_prediction,)
    segmentation_tab.widget_layer_select.prediction.value = napari_prediction
    segmentation_tab.widget_layer_select.superpixels.choices = (napari_segmentation,)
    segmentation_tab.widget_layer_select.superpixels.value = napari_segmentation

    segmentation_tab.widget_agglomeration(
        mode=segmentation_tab.AGGLOMERATION_MODES[0][1]
    )
    mocked_scheduler.assert_called_once()
    assert clustering_segmentation_task == mocked_scheduler.call_args[0][0]
    mocked_log.assert_not_called()

    mocked_log.reset_mock()
    mocked_scheduler.reset_mock()

    segmentation_tab.widget_agglomeration(
        mode=segmentation_tab.AGGLOMERATION_MODES[1][1]
    )
    mocked_scheduler.assert_called_once()
    assert clustering_segmentation_task == mocked_scheduler.call_args[0][0]
    mocked_log.assert_not_called()

    mocked_log.reset_mock()
    mocked_scheduler.reset_mock()

    segmentation_tab.widget_agglomeration(
        mode=segmentation_tab.AGGLOMERATION_MODES[2][1]
    )
    mocked_scheduler.assert_called_once()
    assert clustering_segmentation_task == mocked_scheduler.call_args[0][0]
    mocked_log.assert_not_called()


def test_agglomeration_lmc(
    segmentation_tab, mocker, napari_raw, napari_prediction, napari_segmentation
):
    mocked_scheduler = mocker.patch(
        target="plantseg.viewer_napari.widgets.segmentation.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="plantseg.viewer_napari.widgets.segmentation.log",
        autospec=True,
    )
    segmentation_tab.widget_layer_select.prediction.choices = (napari_prediction,)
    segmentation_tab.widget_layer_select.prediction.value = napari_prediction
    segmentation_tab.widget_layer_select.superpixels.choices = (napari_segmentation,)
    segmentation_tab.widget_layer_select.superpixels.value = napari_segmentation
    segmentation_tab.widget_layer_select.nuclei.choices = (napari_raw,)
    segmentation_tab.widget_layer_select.nuclei.value = napari_raw
    segmentation_tab.widget_agglomeration(
        mode=segmentation_tab.AGGLOMERATION_MODES[3][1]
    )
    mocked_scheduler.assert_called_once()
    assert lmc_segmentation_task == mocked_scheduler.call_args[0][0]
    mocked_log.assert_not_called()


def test_agglomeration_lmc_missing(
    segmentation_tab, mocker, napari_raw, napari_prediction, napari_segmentation
):
    mocked_scheduler = mocker.patch(
        target="plantseg.viewer_napari.widgets.segmentation.schedule_task",
        autospec=True,
    )
    mocked_log = mocker.patch(
        target="plantseg.viewer_napari.widgets.segmentation.log",
        autospec=True,
    )
    segmentation_tab.widget_layer_select.prediction.choices = (napari_prediction,)
    segmentation_tab.widget_layer_select.prediction.value = napari_prediction
    segmentation_tab.widget_layer_select.superpixels.choices = (napari_segmentation,)
    segmentation_tab.widget_layer_select.superpixels.value = napari_segmentation
    segmentation_tab.widget_layer_select.nuclei.choices = (None,)
    segmentation_tab.widget_layer_select.nuclei.value = None
    segmentation_tab.widget_agglomeration(
        mode=segmentation_tab.AGGLOMERATION_MODES[3][1]
    )
    mocked_scheduler.assert_not_called()
    mocked_log.assert_called_with(
        "Nuclei must be an Image or Labels layer",
        thread="Segmentation",
        level="WARNING",
    )


def test_on_mode_changed(segmentation_tab, mocker):
    mocked_show = mocker.patch.object(
        segmentation_tab.widget_layer_select.nuclei, "show"
    )
    mocked_hide = mocker.patch.object(
        segmentation_tab.widget_layer_select.nuclei, "hide"
    )

    segmentation_tab.widget_agglomeration.mode.value = (
        segmentation_tab.AGGLOMERATION_MODES[0][1]
    )
    mocked_show.assert_not_called()
    mocked_hide.assert_not_called()
    mocked_show.reset_mock()
    mocked_hide.reset_mock()

    segmentation_tab.widget_agglomeration.mode.value = (
        segmentation_tab.AGGLOMERATION_MODES[1][1]
    )
    mocked_show.assert_not_called()
    mocked_hide.assert_called_once()
    mocked_show.reset_mock()
    mocked_hide.reset_mock()
    segmentation_tab.widget_agglomeration.mode.value = (
        segmentation_tab.AGGLOMERATION_MODES[2][1]
    )
    mocked_show.assert_not_called()
    mocked_hide.assert_called_once()
    mocked_show.reset_mock()
    mocked_hide.reset_mock()
    segmentation_tab.widget_agglomeration.mode.value = (
        segmentation_tab.AGGLOMERATION_MODES[3][1]
    )
    mocked_show.assert_called_once()
    mocked_hide.assert_not_called()
    mocked_show.reset_mock()
    mocked_hide.reset_mock()


def test_on_show_advanced_changed(segmentation_tab):
    segmentation_tab.widget_dt_ws.show_advanced.value = True
    segmentation_tab.widget_dt_ws.show_advanced.value = False


def test_update_layer_selection(
    segmentation_tab,
    mocker,
    napari_raw,
    napari_prediction,
    napari_segmentation,
    make_napari_viewer_proxy,
):
    viewer = make_napari_viewer_proxy()
    viewer.add_layer(napari_raw)

    sentinel = mocker.sentinel
    sentinel.value = napari_raw
    sentinel.type = "inserted"
    assert segmentation_tab.widget_layer_select.layer.value is None
    assert segmentation_tab.widget_layer_select.nuclei.value is None
    assert segmentation_tab.widget_layer_select.prediction.value is None
    assert segmentation_tab.widget_layer_select.superpixels.value is None

    segmentation_tab.update_layer_selection(sentinel)
    assert segmentation_tab.widget_layer_select.layer.value == sentinel.value
    assert segmentation_tab.widget_layer_select.nuclei.value == sentinel.value

    viewer.add_layer(napari_prediction)
    sentinel.value = napari_prediction
    sentinel.type = "inserted"
    segmentation_tab.update_layer_selection(sentinel)
    assert segmentation_tab.widget_layer_select.prediction.value == sentinel.value

    viewer.add_layer(napari_segmentation)
    sentinel.value = napari_segmentation
    sentinel.type = "inserted"
    segmentation_tab.update_layer_selection(sentinel)
    assert segmentation_tab.widget_layer_select.superpixels.value == sentinel.value
