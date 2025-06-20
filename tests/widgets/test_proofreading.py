import pytest
from plantseg.viewer_napari.widgets import proofreading


def test_copy_if_not_none(mocker):
    mock = mocker.patch.object(mocker.sentinel, "copy")
    assert proofreading.copy_if_not_none(mocker.sentinel)
    mock.assert_called_once()

    assert proofreading.copy_if_not_none(None) is None
