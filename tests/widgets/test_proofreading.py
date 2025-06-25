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

        mocks = mocker.patch.multiple(
            "plantseg.viewer_napari.widgets.proofreading.ProofreadingHandler",
            # corrected_cells_mask=napari_segmentation.data,
            corrected_cells_mask=[4],
            corrected_cells=set((1, 2, 3)),
            scale=mocker.DEFAULT,
        )
        mock_update_layer = mocker.patch(
            "plantseg.viewer_napari.widgets.proofreading.update_layer",
        )

        proof.save_state_to_disk(
            filepath=h5_path, raw=napari_raw, pmap=napari_prediction
        )

        with h5py.File(h5_path, "r") as f:
            assert all([k in f.keys() for k in ("label", "mask", "pmap", "raw")])

        proof.load_state_from_disk(h5_path)

        mock_update_layer.assert_called_with(
            [4],
            proofreading.CORRECTED_CELLS_LAYER_NAME,
            scale=mocks["scale"],
            colormap=proofreading.correct_cells_cmap,
            opacity=1,
        )

    def test_toggle_corrected_cell(self, mocker, proof):
        mock = mocker.patch.object(proof._state, "corrected_cells")
        proof._toggle_corrected_cell(0)
        mock.add.assert_called_with(0)

        mock.__contains__ = lambda a, b: True

        proof._toggle_corrected_cell(0)
        mock.remove.assert_called_with(0)

    def test_update_masks(self, proof, mocker):
        mocks = mocker.patch.multiple(
            "plantseg.viewer_napari.widgets.proofreading",
            update_corrected_cells_mask_layer=mocker.DEFAULT,
            get_layer_data=mocker.DEFAULT,
        )
        mocks_handler = mocker.patch.multiple(
            "plantseg.viewer_napari.widgets.proofreading.ProofreadingHandler",
            scale=mocker.DEFAULT,
            segmentation=mocker.DEFAULT,
        )
        proof._update_masks(0)
        mocks["update_corrected_cells_mask_layer"].assert_called_once()

    def test_toggle_corrected_cell_(self, mocker, proof):
        mocks = mocker.patch.multiple(
            "plantseg.viewer_napari.widgets.proofreading.ProofreadingHandler",
            _toggle_corrected_cell=mocker.DEFAULT,
            _update_masks=mocker.DEFAULT,
        )
        proof.toggle_corrected_cell(mocker.sentinel)
        [m.assert_called_with(mocker.sentinel) for m in mocks.values()]

    def test_update_corrected_cells_mask_slice_to_viewer(self, mocker, proof):
        mocks = mocker.patch.multiple(
            "plantseg.viewer_napari.widgets.proofreading",
            update_region=mocker.DEFAULT,
            preserve_labels=mocker.DEFAULT,
        )
        mocker.patch(
            "plantseg.viewer_napari.widgets.proofreading.ProofreadingHandler.scale",
        )

        proof.update_corrected_cells_mask_slice_to_viewer(
            mocker.sentinel, mocker.sentinel
        )
        [m.assert_called_once() for m in mocks.values()]

    def test_update_after_proofreading(self, mocker, proof):
        mocks = mocker.patch.multiple(
            "plantseg.viewer_napari.widgets.proofreading.ProofreadingHandler",
            create=True,
            _state=mocker.DEFAULT,
            seg_layer_name=mocker.DEFAULT,
            scale=mocker.DEFAULT,
        )
        mocker.patch(
            "plantseg.viewer_napari.widgets.proofreading.update_region",
        )
        proof.update_after_proofreading(
            mocker.sentinel, mocker.sentinel, mocker.sentinel
        )


def test_widget_clean_scribble(mocker, make_napari_viewer_proxy, napari_raw):
    mock = mocker.patch(
        "plantseg.viewer_napari.widgets.proofreading.segmentation_handler.reset_scribbles",
    )
    mock_log = mocker.patch(
        "plantseg.viewer_napari.widgets.proofreading.log",
    )
    viewer = make_napari_viewer_proxy()

    proofreading.widget_clean_scribble(viewer)
    mock_log.assert_called_with(
        "Proofreading widget not initialized. Run the proofreading widget tool once first",
        thread="Clean scribble",
    )
    mock_log.reset_mock()

    proofreading.segmentation_handler._state.active = True
    proofreading.widget_clean_scribble(viewer)
    mock_log.assert_called_with(
        "Scribble Layer not defined. Run the proofreading widget tool once first",
        thread="Clean scribble",
    )
    mock_log.reset_mock()

    viewer.add_layer(napari_raw)
    viewer.layers[0].name = "Scribbles"
    proofreading.widget_clean_scribble(viewer)
    mock.assert_called_once()


def test_widget_add_label_to_corrected(
    mocker, make_napari_viewer_proxy, napari_segmentation
):
    viewer = make_napari_viewer_proxy()
    with pytest.raises(ValueError):
        proofreading.widget_add_label_to_corrected(viewer, (0, 0, 0))

    mocker.patch(
        "plantseg.viewer_napari.widgets.proofreading.segmentation_handler._scale",
        (1, 1, 1),
    )
    mocker.patch(
        "plantseg.viewer_napari.widgets.proofreading.segmentation_handler._state.current_seg_layer_name",
        proofreading.CORRECTED_CELLS_LAYER_NAME,
    )
    mock = mocker.patch(
        "plantseg.viewer_napari.widgets.proofreading.segmentation_handler.toggle_corrected_cell",
    )

    napari_segmentation.name = proofreading.CORRECTED_CELLS_LAYER_NAME
    viewer.add_layer(napari_segmentation)
    proofreading.widget_add_label_to_corrected(viewer, (0, 0, 0))
    mock.assert_called_once()


def test_initialize_proofreading(mocker, napari_segmentation, make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_layer(napari_segmentation)
    pi = PlantSegImage.from_napari_layer(napari_segmentation)
    mock = mocker.patch(
        "plantseg.viewer_napari.widgets.proofreading.setup_proofreading_widget",
    )
    proofreading.initialize_proofreading(pi)
    mock.assert_called_once()


def test_initialize_from_layer_wrong_layer(mocker, napari_segmentation):
    napari_segmentation.name = proofreading.SCRIBBLES_LAYER_NAME
    mock = mocker.patch(
        "plantseg.viewer_napari.widgets.proofreading.log",
    )
    proofreading.initialize_from_layer(napari_segmentation, False)
    mock.assert_called_with(
        "Scribble or corrected cells layer is not intended to be proofread, choose a segmentation",
        thread="Proofreading tool",
        level="error",
    )


def test_initialize_from_layer_not_sure(mocker, napari_segmentation):
    mock = mocker.patch(
        "plantseg.viewer_napari.widgets.proofreading.log",
    )
    proofreading.segmentation_handler._state.active = True
    proofreading.initialize_from_layer(napari_segmentation, False)
    mock.assert_called_with(
        "Proofreading is already initialized. Are you sure you want to reset everything?",
        thread="Proofreading tool",
        level="warning",
    )


def test_initialize_from_layer(mocker, napari_segmentation, make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    mocks = mocker.patch.multiple(
        "plantseg.viewer_napari.widgets.proofreading",
        initialize_proofreading=mocker.DEFAULT,
        widget_proofreading_initialisation=mocker.DEFAULT,
    )
    proofreading.segmentation_handler._state.active = True
    proofreading.initialize_from_layer(napari_segmentation, True)
    mocks["initialize_proofreading"].assert_called_once()
    mocks["widget_proofreading_initialisation"].are_you_sure.hide.assert_called_once()


def test_initialize_from_file_not_sure(mocker):
    mock = mocker.patch(
        "plantseg.viewer_napari.widgets.proofreading.log",
    )
    proofreading.segmentation_handler._state.active = True
    proofreading.initialize_from_file(Path(), False)
    mock.assert_called_with(
        "Proofreading is already initialized. Are you sure you want to reset everything?",
        thread="Proofreading tool",
        level="warning",
    )


def test_initialize_from_file(mocker, make_napari_viewer_proxy):
    mocks = mocker.patch.multiple(
        "plantseg.viewer_napari.widgets.proofreading",
        segmentation_handler=mocker.DEFAULT,
        setup_proofreading_widget=mocker.DEFAULT,
    )
    proofreading.segmentation_handler._state.active = True
    proofreading.initialize_from_file(mocker.sentinel, True)
    mocks["segmentation_handler"].load_state_from_disk.assert_called_with(
        mocker.sentinel
    )
    mocks["setup_proofreading_widget"].assert_called_once()


def test_widget_proofreading_initialisation(mocker, napari_segmentation):
    mocks = mocker.patch.multiple(
        "plantseg.viewer_napari.widgets.proofreading",
        log=mocker.DEFAULT,
        initialize_from_layer=mocker.DEFAULT,
        initialize_from_file=mocker.DEFAULT,
    )
    proofreading.widget_proofreading_initialisation("New", None, None)
    mocks["log"].assert_called_with(
        "No segmentation layer selected",
        thread="Proofreading tool",
        level="error",
    )
    proofreading.widget_proofreading_initialisation("Continue", None, None)
    mocks["log"].assert_called_with(
        "No state file selected", thread="Proofreading tool", level="error"
    )

    mocks["initialize_from_layer"].assert_not_called()
    mocks["initialize_from_file"].assert_not_called()

    proofreading.widget_proofreading_initialisation(
        "New", napari_segmentation, None, mocker.sentinel
    )
    mocks["initialize_from_layer"].assert_called_with(
        napari_segmentation, are_you_sure=mocker.sentinel
    )
    mocks["initialize_from_file"].assert_not_called()
    mocks["initialize_from_layer"].reset_mock()

    proofreading.widget_proofreading_initialisation(
        "Continue", None, Path(), mocker.sentinel
    )
    mocks["initialize_from_file"].assert_called_with(
        Path(), are_you_sure=mocker.sentinel
    )
    mocks["initialize_from_layer"].assert_not_called()


def test_on_mode_change(mocker):
    mock = mocker.patch(
        "plantseg.viewer_napari.widgets.proofreading.widget_proofreading_initialisation"
    )
    proofreading._on_mode_changed("New")
    mock.segmentation.show.assert_called_once()
    mock.filepath.hide.assert_called_once()
    proofreading._on_mode_changed("Continue")
    mock.segmentation.hide.assert_called_once()
    mock.filepath.show.assert_called_once()


def test_widget_split_and_merge_from_scribbles_log(mocker, napari_raw):
    mocks = mocker.patch.multiple(
        "plantseg.viewer_napari.widgets.proofreading",
        log=mocker.DEFAULT,
        segmentation_handler=mocker.DEFAULT,
    )
    mocks["segmentation_handler"].active = False
    proofreading.widget_split_and_merge_from_scribbles(mocker.sentinel, napari_raw)
    mocks["log"].assert_called_with(
        "Proofreading is not initialized. Run the initialization widget first.",
        thread="Proofreading tool",
    )
    mocks["segmentation_handler"].active = True
    proofreading.widget_split_and_merge_from_scribbles(mocker.sentinel, None)
    mocks["log"].assert_called_with(
        "Please select a boundary image first!",
        thread="Proofreading tool",
    )


def test_widget_split_and_merge_from_scribbles(mocker, napari_raw):
    proofreading.segmentation_handler._state.active = True
    mocks = mocker.patch.multiple(
        "plantseg.viewer_napari.widgets.proofreading",
        segmentation_handler=mocker.DEFAULT,
        split_merge_from_seeds=mocker.DEFAULT,
    )
    proofreading.widget_split_and_merge_from_scribbles(mocker.sentinel, napari_raw)
    mocks["split_merge_from_seeds"].assert_not_called()

    mock_sum = mocks["segmentation_handler"].scribbles.sum = mocker.Mock()
    mock_sum.return_value = 5
    mock_lock = mocks["segmentation_handler"].is_locked = mocker.Mock()
    mock_lock.return_value = False
    mocks["split_merge_from_seeds"].return_value = [mocker.sentinel] * 3

    proofreading.widget_split_and_merge_from_scribbles(mocker.sentinel, napari_raw)
    mocks["segmentation_handler"].is_locked.assert_called_once()
    mock_sum.assert_called_once()
    mocks["split_merge_from_seeds"].assert_called_once()
    mocks["segmentation_handler"].update_after_proofreading.assert_called_with(
        *[mocker.sentinel] * 3
    )


def test_widget_filter_segmentation_log(mocker):
    mocks = mocker.patch.multiple(
        "plantseg.viewer_napari.widgets.proofreading",
        log=mocker.DEFAULT,
        segmentation_handler=mocker.DEFAULT,
    )
    mocks["segmentation_handler"].active = False
    with pytest.raises(ValueError):
        proofreading.widget_filter_segmentation()
    mocks["log"].assert_called_with(
        "Proofreading widget not initialized. Run the proofreading widget tool once first",
        thread="Export correct labels",
        level="error",
    )


def test_widget_filter_segmentation(mocker):
    mocks = mocker.patch.multiple(
        "plantseg.viewer_napari.widgets.proofreading",
        segmentation_handler=mocker.DEFAULT,
        split_merge_from_seeds=mocker.DEFAULT,
        ImageProperties=mocker.DEFAULT,
        PlantSegImage=mocker.DEFAULT,
        napari=mocker.DEFAULT,
    )

    mock_lock = mocks["segmentation_handler"].is_locked = mocker.Mock()
    mock_lock.return_value = False
    proofreading.widget_filter_segmentation()


def test_widget_undo(mocker):
    mocks = mocker.patch.multiple(
        "plantseg.viewer_napari.widgets.proofreading",
        segmentation_handler=mocker.DEFAULT,
        log=mocker.DEFAULT,
    )
    mocks["segmentation_handler"].active = False
    proofreading.widget_undo()
    mocks["log"].assert_called_with(
        "Proofreading widget not initialized. Nothing to undo.", thread="Undo"
    )
    mocks["segmentation_handler"].active = True
    proofreading.widget_undo()
    mocks["segmentation_handler"].undo.assert_called_once()


def test_widget_redo(mocker):
    mocks = mocker.patch.multiple(
        "plantseg.viewer_napari.widgets.proofreading",
        segmentation_handler=mocker.DEFAULT,
        log=mocker.DEFAULT,
    )
    mocks["segmentation_handler"].active = False
    proofreading.widget_redo()
    mocks["log"].assert_called_with(
        "Proofreading widget not initialized. Nothing to redo.", thread="Redo"
    )
    mocks["segmentation_handler"].active = True
    proofreading.widget_redo()
    mocks["segmentation_handler"].redo.assert_called_once()


def test_widget_save_state(mocker):
    mocks = mocker.patch.multiple(
        "plantseg.viewer_napari.widgets.proofreading",
        segmentation_handler=mocker.DEFAULT,
    )
    proofreading.widget_save_state(mocker.sentinel)
    mocks["segmentation_handler"].save_state_to_disk.assert_called_with(
        mocker.sentinel,
        raw=None,
        pmap=None,
    )


def test_setup_keybindings(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    proofreading.setup_proofreading_keybindings()


def test_setup_proofreading_widget():
    proofreading.setup_proofreading_widget()
