from pathlib import Path

import numpy as np
import pytest

from plantseg.core.image import ImageType
from plantseg.viewer_napari.widgets.input import Input_Tab, PathMode


@pytest.fixture
def input_tab():
    """Fixture to create an Input_Tab instance for testing"""
    return Input_Tab()


def test_input_tab_initialization(input_tab):
    container = input_tab.get_container()

    assert len(container) == 8


def test_input_tab_open_file(input_tab, mocker):
    mocked_scheduler = mocker.patch(
        target="plantseg.viewer_napari.widgets.input.schedule_task",
        autospec=True,
    )
    kwargs = {
        "path_mode": True,
        "path": Path(),
        "dataset_key": "",
        "button_key_refresh": True,
        "stack_layout": "",
        "layer_type": ImageType.IMAGE.value,
        "new_layer_name": "",
        "update_other_widgets": False,
    }
    input_tab.widget_open_file(**kwargs)
    mocked_scheduler.assert_not_called()

    input_tab.path_changed_once = True
    input_tab.widget_open_file(**kwargs)
    mocked_scheduler.assert_called_once()


def test_open_file_widget_path_handling(input_tab):
    assert input_tab.widget_open_file.path.mode.value == "r"
    assert input_tab.widget_open_file.path.label == "File path"

    input_tab._on_path_mode_changed(PathMode.FILE.value)
    assert input_tab.widget_open_file.path.mode.value == "r"
    assert input_tab.widget_open_file.path.label == "File path"

    input_tab._on_path_mode_changed(PathMode.DIR.value)
    assert input_tab.widget_open_file.path.mode.value == "d"
    assert input_tab.widget_open_file.path.label == "Zarr path\n(.zarr)"


def test_look_up_dataset_keys_empty(input_tab, zarr_file_empty, mocker):
    mock_show = mocker.patch.object(input_tab.widget_open_file.dataset_key, "show")
    mock_hide = mocker.patch.object(input_tab.widget_open_file.dataset_key, "hide")

    input_tab.look_up_dataset_keys(zarr_file_empty)
    mock_show.assert_called_once()
    mock_hide.assert_called_once()


def test_look_up_dataset_keys_3d(input_tab, zarr_file_3d, mocker):
    mock_show = mocker.patch.object(input_tab.widget_open_file.dataset_key, "show")
    mock_hide = mocker.patch.object(input_tab.widget_open_file.dataset_key, "hide")

    input_tab.look_up_dataset_keys(zarr_file_3d)
    mock_show.assert_called_once()
    mock_hide.assert_not_called()
    assert input_tab.widget_open_file.dataset_key.value == "raw"
    assert all(
        n in ("raw", "raw_2") for n in input_tab.widget_open_file.dataset_key.choices
    )


def test_set_voxel_size(input_tab, napari_image, mocker):
    input_tab.widget_details_layer_select.layer.choices = [napari_image]
    input_tab.widget_details_layer_select.layer.value = napari_image

    mocked_scheduler = mocker.patch(
        target="plantseg.viewer_napari.widgets.input.schedule_task",
        autospec=True,
    )

    input_tab.widget_set_voxel_size(input_tab, (1.0, 1.0, 1.0))
    mocked_scheduler.assert_called_once()
