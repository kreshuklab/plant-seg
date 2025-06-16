from pathlib import Path

import pytest

from plantseg.core.image import ImageType
from plantseg.viewer_napari.widgets.preprocessing import Preprocessing_Tab, RescaleType


@pytest.fixture
def preprocessing_tab():
    return Preprocessing_Tab()


def test_preprocessing_tab_initialization(preprocessing_tab):
    container = preprocessing_tab.get_container()
    assert len(container) == 13


def test_on_layer_select_raw(preprocessing_tab, napari_raw, mocker):
    mocked_hide = mocker.patch.object(
        preprocessing_tab.widget_rescaling.rescaling_factor[0], "hide"
    )
    mocked_show = mocker.patch.object(
        preprocessing_tab.widget_rescaling.rescaling_factor[0], "show"
    )

    preprocessing_tab._on_layer_selection(napari_raw)

    mocked_show.assert_called_once()
    mocked_hide.assert_not_called()


def test_on_layer_select_label(preprocessing_tab, napari_labels):
    assert preprocessing_tab.widget_rescaling.order.value == 0
    preprocessing_tab.widget_rescaling.order.value = 1

    preprocessing_tab._on_layer_selection(napari_labels)
    assert preprocessing_tab.widget_rescaling.order.value == 0


def test_on_layer_select_no_meta(preprocessing_tab, napari_shapes):
    with pytest.raises(ValueError):
        preprocessing_tab._on_layer_selection(napari_shapes)
