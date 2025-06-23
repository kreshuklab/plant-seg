from collections import deque
from pathlib import Path

import h5py
import numpy as np
import pytest

from plantseg.core.image import PlantSegImage
from plantseg.viewer_napari.widgets import proofreading


def test_copy_if_not_none(mocker):
    mock = mocker.patch.object(mocker.sentinel, "copy")
    assert proofreading.copy_if_not_none(mocker.sentinel)
    mock.assert_called_once()

    assert proofreading.copy_if_not_none(None) is None


def test_get_current_viewer(make_napari_viewer_proxy):
    assert proofreading.get_current_viewer_wrapper() is None
    viewer = make_napari_viewer_proxy()
    assert proofreading.get_current_viewer_wrapper() == viewer


def test_update_layer_empty(make_napari_viewer_proxy, napari_labels):
    viewer = make_napari_viewer_proxy()
    proofreading.update_layer(
        napari_labels.data, layer_name="test", scale=napari_labels.scale
    )
    np.testing.assert_array_equal(viewer.layers[0].data, napari_labels.data)
    np.testing.assert_array_equal(viewer.layers[0].scale, napari_labels.scale)
    assert viewer.layers[0].name == "test"


def test_update_layer(make_napari_viewer_proxy, napari_labels):
    viewer = make_napari_viewer_proxy()
    viewer.add_labels(np.zeros((5, 5, 5), dtype=int), name="test", scale=[1, 1, 1])
    proofreading.update_layer(
        napari_labels.data, layer_name="test", scale=napari_labels.scale
    )
    np.testing.assert_array_equal(viewer.layers[0].data, napari_labels.data)
    np.testing.assert_array_equal(viewer.layers[0].scale, napari_labels.scale)
    assert viewer.layers[0].name == "test"


def test_updat_corrected_cells_mask_layer(mocker):
    mock = mocker.patch("plantseg.viewer_napari.widgets.proofreading.update_layer")
    proofreading.update_corrected_cells_mask_layer(
        data=mocker.sentinel, scale=mocker.sentinel
    )

    mock.assert_called_with(
        mocker.sentinel,
        proofreading.CORRECTED_CELLS_LAYER_NAME,
        scale=mocker.sentinel,
        colormap=proofreading.correct_cells_cmap,
        opacity=1,
    )


def test_update_scribbles_layer(mocker):
    mock = mocker.patch("plantseg.viewer_napari.widgets.proofreading.update_layer")
    proofreading.update_scribbles_layer(data=mocker.sentinel, scale=mocker.sentinel)
    mock.assert_called_with(
        mocker.sentinel, proofreading.SCRIBBLES_LAYER_NAME, scale=mocker.sentinel
    )


def test_update_region_empty(mocker, make_napari_viewer_proxy, napari_raw):
    viewer = make_napari_viewer_proxy()
    with pytest.raises(ValueError):
        proofreading.update_region(
            napari_raw.data,
            layer_name="test",
            region_slice=(slice(0, 5), slice(0, 5), slice(0, 5)),
            scale=napari_raw.scale,
        )


def test_update_region(make_napari_viewer_proxy, napari_labels, napari_raw):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(napari_raw.data, name="test", scale=[1, 1, 1])
    sl = (slice(0, 2), slice(0, 2), slice(0, 2))
    proofreading.update_region(
        napari_labels.data[sl],
        layer_name="test",
        region_slice=sl,
        scale=napari_labels.scale,
    )
    np.testing.assert_array_equal(viewer.layers[0].data[sl], napari_labels.data[sl])
    np.testing.assert_array_equal(
        viewer.layers[0].data[(slice(2, -1), slice(2, -1), slice(2, -1))],
        napari_raw.data[
            (
                slice(2, -1),
                slice(2, -1),
                slice(2, -1),
            )
        ],
    )
    np.testing.assert_array_equal(viewer.layers[0].scale, napari_labels.scale)
    assert viewer.layers[0].name == "test"


def test_get_layer_data_empty(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    with pytest.raises(ValueError):
        proofreading.get_layer_data("test")


def test_get_layer_data(make_napari_viewer_proxy, napari_raw):
    viewer = make_napari_viewer_proxy()
    viewer.add_layer(napari_raw)
    assert np.all(proofreading.get_layer_data("test_image") == napari_raw.data)


def test_get_layer_region_data_empty(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    with pytest.raises(ValueError):
        proofreading.get_layer_region_data("test", (slice(0, 1),))


def test_get_layer_region_data(make_napari_viewer_proxy, napari_raw):
    viewer = make_napari_viewer_proxy()
    viewer.add_layer(napari_raw)
    sl = (slice(0, 2), slice(0, 2), slice(0, 2))
    assert np.all(
        proofreading.get_layer_region_data("test_image", sl) == napari_raw.data[sl]
    )


def test_preserve_labels(make_napari_viewer_proxy, napari_raw):
    viewer = make_napari_viewer_proxy()
    viewer.add_layer(napari_raw)
    proofreading.preserve_labels("test_image")
    assert viewer.layers["test_image"].preserve_labels


@pytest.fixture
def proof():
    return proofreading.ProofreadingHandler()


class TestProofreadingHandler:
    def test_init(self, proof):
        proof

    def test_reset_scribbles(self, mocker, proof):
        mock = mocker.patch("plantseg.viewer_napari.widgets.proofreading.update_layer")
        proof.reset_scribbles()
        mock.assert_not_called()

        proof._state.active = True
        mocker.patch.multiple(
            "plantseg.viewer_napari.widgets.proofreading.ProofreadingHandler",
            segmentation=mocker.DEFAULT,
            scale=mocker.DEFAULT,
        )
        proof.reset_scribbles()
        mock.assert_called_once()

    def test_reset_corrected(self, mocker, proof):
        mock = mocker.patch("plantseg.viewer_napari.widgets.proofreading.update_layer")
        proof.reset_corrected()
        mock.assert_not_called()

        proof._state.active = True
        mocker.patch.multiple(
            "plantseg.viewer_napari.widgets.proofreading.ProofreadingHandler",
            segmentation=mocker.DEFAULT,
            scale=mocker.DEFAULT,
        )
        proof.reset_corrected()
        mock.assert_called_once()

    def test_bboxes(self, mocker, proof):
        mock = mocker.patch.object(proof, "reset_bboxes")
        with pytest.raises(AssertionError):
            proof.bboxes
        mock.assert_called_once()

    def test_reset_bboxes(self, mocker, proof):
        mock = mocker.patch("plantseg.viewer_napari.widgets.proofreading.get_bboxes")
        with pytest.raises(ValueError):
            proof.reset_bboxes()
        mock.assert_not_called()

        proof._state.active = True
        mocker.patch(
            "plantseg.viewer_napari.widgets.proofreading.ProofreadingHandler.segmentation",
            new_callable=mocker.PropertyMock,
        )
        proof.reset_bboxes()
        mock.assert_called_once()

    def test_reset(self, proof):
        proof.reset()

    def test_setup(self, proof, mocker, napari_segmentation):
        mock_reset = mocker.patch.object(proof, "reset")
        mock_reset_bboxes = mocker.patch.object(proof, "reset_bboxes")
        mock_reset_corrected = mocker.patch.object(proof, "reset_corrected")
        mock_reset_scribbles = mocker.patch.object(proof, "reset_scribbles")
        proof.setup(PlantSegImage.from_napari_layer(napari_segmentation))
        mock_reset.assert_called_once()
        mock_reset_bboxes.assert_called_once()
        mock_reset_corrected.assert_called_once()
        mock_reset_scribbles.assert_called_once()

    def test_capture_state(self, proof, mocker):
        mocker.patch.multiple(
            "plantseg.viewer_napari.widgets.proofreading.ProofreadingHandler",
            segmentation=mocker.DEFAULT,
            corrected_cells=mocker.DEFAULT,
            corrected_cells_mask=mocker.DEFAULT,
            bboxes=mocker.DEFAULT,
        )
        proof._capture_state()

    def test_save_to_history(self, proof, mocker):
        mock = mocker.patch.object(proof, "_capture_state")
        mock.return_value = mocker.sentinel
        proof.save_to_history()
        assert proof._state.history_undo[-1] == mocker.sentinel

    def test_restore_state(self, proof, mocker):
        mock_update_layer = mocker.patch(
            "plantseg.viewer_napari.widgets.proofreading.update_layer",
        )
        mocker.patch.multiple(
            "plantseg.viewer_napari.widgets.proofreading.ProofreadingHandler",
            reset_scribbles=mocker.DEFAULT,
            seg_layer_name=mocker.DEFAULT,
            scale=mocker.DEFAULT,
        )
        proof._restore_state(mocker.sentinel)

        assert mock_update_layer.call_count == 2

    def test__perform_undo_redo(self, proof, mocker):
        mock = mocker.patch.multiple(
            "plantseg.viewer_napari.widgets.proofreading.ProofreadingHandler",
            _capture_state=mocker.DEFAULT,
            _restore_state=mocker.DEFAULT,
        )
        proof._perform_undo_redo(deque(), deque(), "smth")
        [m.assert_not_called() for m in mock.values()]

        mock["_capture_state"].return_value = "current"
        pop = deque(("history",))
        append = deque()
        proof._perform_undo_redo(pop, append, "smth")
        mock["_restore_state"].assert_called_with("history")
        assert append == deque(("current",))

    def test_undo(self, proof, mocker):
        mock = mocker.patch(
            "plantseg.viewer_napari.widgets.proofreading.ProofreadingHandler._perform_undo_redo"
        )
        proof.undo()
        mock.assert_called_once()

    def test_redo(self, proof, mocker):
        mock = mocker.patch(
            "plantseg.viewer_napari.widgets.proofreading.ProofreadingHandler._perform_undo_redo"
        )
        proof.redo()
        mock.assert_called_once()

    def test_save_state_to_disk_suffix(self, proof, mocker):
        mock = mocker.patch("plantseg.viewer_napari.widgets.proofreading.log")
        proof.save_state_to_disk(Path("not_h5.file"), raw=None, pmap=None)
        mock.assert_called_once()

    def test_save_state_to_disk_load(
        self,
        proof,
        mocker,
        tmp_path,
        napari_raw,
        napari_prediction,
        napari_segmentation,
        make_napari_viewer_proxy,
    ):
        h5_path = tmp_path / "valid.h5"
        viewer = make_napari_viewer_proxy()
        viewer.add_layer(napari_segmentation)
        proof._state.current_seg_layer_name = "test_image"

        mocker.patch.multiple(
            "plantseg.viewer_napari.widgets.proofreading.ProofreadingHandler",
            corrected_cells_mask=napari_segmentation.data,
            corrected_cells=set((1, 2, 3)),
        )

        proof.save_state_to_disk(
            filepath=h5_path, raw=napari_raw, pmap=napari_prediction
        )

        with h5py.File(h5_path, "r") as f:
            assert all([k in f.keys() for k in ("label", "mask", "pmap", "raw")])

        proof.load_state_from_disk(h5_path)
