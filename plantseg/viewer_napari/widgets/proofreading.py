from concurrent.futures import Future
from typing import Union

import napari
import numpy as np
from magicgui import magicgui
from napari.layers import Image, Labels
from napari.qt.threading import thread_worker
from napari.types import LayerDataTuple

from plantseg.functionals.proofreading.split_merge_tools import split_merge_from_seeds
from plantseg.functionals.proofreading.utils import get_bboxes
from plantseg.plantseg_image import PlantSegImage, SemanticType
from plantseg.viewer_napari.logging import napari_formatted_logging

DEFAULT_KEY_BINDING_PROOFREAD = 'n'
DEFAULT_KEY_BINDING_CLEAN = 'b'
SCRIBBLES_LAYER_NAME = 'Scribbles'
CORRECTED_CELLS_LAYER_NAME = 'Correct Labels'


class ProofreadingHandler:
    _status: bool
    _current_seg_layer_name: Union[str, None]
    _corrected_cells: set
    _segmentation: Union[np.ndarray, None]
    _current_seg_properties: Union[dict, None]
    _corrected_cells_mask: Union[np.ndarray, None]
    _scribbles: Union[np.ndarray, None]
    _bboxes: Union[np.ndarray, None]

    _lock: bool = False
    scale: Union[tuple, None] = None
    scribbles_layer_name = SCRIBBLES_LAYER_NAME
    corrected_cells_layer_name = CORRECTED_CELLS_LAYER_NAME
    correct_cells_cmap = {0: None, 1: (0.76388469, 0.02003777, 0.61156412, 1.0)}

    def __init__(self):
        self._status = False

    @property
    def status(self):
        return self._status

    @property
    def seg_layer_name(self):
        return self._current_seg_layer_name

    @property
    def seg_properties(self):
        return self._current_seg_properties

    @property
    def segmentation(self):
        return self._segmentation

    @property
    def scribbles(self):
        return self._scribbles

    @property
    def corrected_cells_mask(self):
        return self._corrected_cells_mask

    @property
    def bboxes(self):
        return self._bboxes

    @property
    def max_label(self):
        return self.segmentation.max()

    @property
    def corrected_cells(self):
        return self._corrected_cells

    def lock(self):
        self._lock = True

    def unlock(self):
        self._lock = False

    def is_locked(self):
        return self._lock

    def setup(self, segmentation: PlantSegImage):
        # make sure all fields are reset
        self.reset()

        self._status = True
        segmentation_data = segmentation.get_data()
        self._current_seg_layer_name = segmentation.name
        self._current_seg_properties = segmentation.properties
        self.scale = segmentation.scale

        self._segmentation = segmentation_data
        self.reset_scribbles()
        self.reset_corrected_cells_mask()

        self._bboxes = get_bboxes(segmentation_data)

    def reset(self):
        self._status = False
        self._current_seg_layer_name = None
        self._corrected_cells = set()

        self._segmentation = None
        self._corrected_cells_mask = None
        self._scribbles = None
        self._bboxes = None
        self.scale = None

    def toggle_corrected_cell(self, cell_id: int):
        self._toggle_corrected_cell(cell_id)
        self._update_masks(cell_id)

    def _toggle_corrected_cell(self, cell_id: int):
        if cell_id in self._corrected_cells:
            self._corrected_cells.remove(cell_id)
        else:
            self._corrected_cells.add(cell_id)

    def _update_masks(self, cell_id: int):
        mask = self._segmentation == cell_id

        self._corrected_cells_mask[mask] += 1
        self._corrected_cells_mask = self._corrected_cells_mask % 2  # act as a toggle

    @staticmethod
    def _update_to_viewer(viewer: napari.Viewer, data: np.ndarray, layer_name: str, **kwargs):
        if layer_name in viewer.layers:
            viewer.layers[layer_name].data = data
            viewer.layers[layer_name].refresh()

        else:
            viewer.add_labels(data, name=layer_name, **kwargs)

    @staticmethod
    def _update_slice_to_viewer(viewer: napari.Viewer, data: np.ndarray, layer_name: str, region_slice):
        if layer_name in viewer.layers:
            viewer.layers[layer_name].data[region_slice] = data
            viewer.layers[layer_name].refresh()
        else:
            raise ValueError(f'Layer {layer_name} not found in viewer')

    def update_scribble_to_viewer(self, viewer: napari.Viewer):
        self._update_to_viewer(viewer, self._scribbles, self.scribbles_layer_name, scale=self.scale)

    def update_scribbles_from_viewer(self, viewer: napari.Viewer):
        self._scribbles = viewer.layers[self.scribbles_layer_name].data

    def reset_scribbles(self):
        self._scribbles = np.zeros_like(self._segmentation).astype(np.uint16)

    def preserve_labels(self, viewer: napari.Viewer, layer_name: str):
        viewer.layers[layer_name].preserve_labels = True
        viewer.layers[layer_name].refresh()

    def update_corrected_cells_mask_to_viewer(self, viewer: napari.Viewer):
        self._update_to_viewer(
            viewer,
            self.corrected_cells_mask,
            self.corrected_cells_layer_name,
            scale=self.scale,
            color=self.correct_cells_cmap,
            opacity=1,
        )
        self.preserve_labels(viewer, self.corrected_cells_layer_name)

    def update_corrected_cells_mask_slice_to_viewer(
        self, viewer: napari.Viewer, slice_data: np.ndarray, region_slice: tuple[slice, ...]
    ):
        self._update_slice_to_viewer(viewer, slice_data, self.corrected_cells_layer_name, region_slice)
        self.preserve_labels(viewer, self.corrected_cells_layer_name)

    def update_after_proofreading(
        self, viewer: napari.Viewer, seg_slice: np.ndarray, region_slice: tuple[slice, ...], bbox: np.ndarray
    ):
        self._bboxes = bbox
        self._update_slice_to_viewer(viewer, seg_slice, self.seg_layer_name, region_slice)

    def reset_corrected_cells_mask(self):
        self._corrected_cells_mask = np.zeros_like(self._segmentation).astype(np.uint16)


segmentation_handler = ProofreadingHandler()


@magicgui(call_button=f'Clean scribbles - < {DEFAULT_KEY_BINDING_CLEAN} >')
def widget_clean_scribble(viewer: napari.Viewer):
    if not segmentation_handler.status:
        napari_formatted_logging(
            'Proofreading widget not initialized. Run the proofreading widget tool once first', thread='Clean scribble'
        )

    if 'Scribbles' not in viewer.layers:
        napari_formatted_logging(
            'Scribble Layer not defined. Run the proofreading widget tool once first', thread='Clean scribble'
        )
        return None

    segmentation_handler.reset_scribbles()
    segmentation_handler.update_scribble_to_viewer(viewer)


widget_clean_scribble.hide()


def widget_add_label_to_corrected(viewer: napari.Viewer, position: tuple[int, ...]):
    if segmentation_handler.corrected_cells_layer_name not in viewer.layers:
        return None

    if len(position) == 2:
        position = [0, *position]

    position = [int(p / s) for p, s in zip(position, segmentation_handler.scale)]
    cell_id = segmentation_handler.segmentation[position[0], position[1], position[2]]
    segmentation_handler.toggle_corrected_cell(cell_id)
    segmentation_handler.update_corrected_cells_mask_to_viewer(viewer)


def initialize_proofreading(viewer: napari.Viewer, segmentation: PlantSegImage) -> bool:
    if segmentation_handler.scribbles_layer_name not in viewer.layers:
        segmentation_handler.reset()

    if segmentation_handler.corrected_cells_layer_name not in viewer.layers:
        segmentation_handler.reset()

    if segmentation_handler.seg_layer_name != segmentation.name:
        segmentation_handler.reset()

    if segmentation_handler.status:
        return False

    segmentation_handler.setup(segmentation)
    segmentation_handler.update_scribble_to_viewer(viewer)
    segmentation_handler.update_corrected_cells_mask_to_viewer(viewer)
    return True


@magicgui(
    call_button='Initialize Proofreading',
    segmentation={'label': 'Segmentation'},
    image={'label': 'Pmap/Image'},
)
def widget_split_and_merge_from_scribbles(
    viewer: napari.Viewer,
    segmentation: Labels,
    image: Image,
) -> None:
    if segmentation is None:
        napari_formatted_logging('Segmentation Layer not defined', thread='Proofreading tool', level='error')
        return None

    if image is None:
        napari_formatted_logging('Image Layer not defined', thread='Proofreading tool', level='error')
        return None

    ps_segmentation = PlantSegImage.from_napari_layer(segmentation)
    ps_image = PlantSegImage.from_napari_layer(image)

    if ps_image.semantic_type == SemanticType.RAW:
        napari_formatted_logging(
            'Pmap/Image layer appears to be a raw image and not a boundary probability map. '
            'For the best proofreading results, try to use a boundaries probability layer '
            '(e.g. from the Run Prediction widget)',
            thread='Proofreading tool',
            level='warning',
        )

    if ps_image.is_multichannel:
        napari_formatted_logging(
            'Pmap/Image layer appears to be a multichannel image. '
            'Proofreading does not support multichannel images. ',
            thread='Proofreading tool',
            level='error',
        )

    image_data = ps_image.get_data()

    if initialize_proofreading(viewer, ps_segmentation):
        napari_formatted_logging('Proofreading initialized', thread='Proofreading tool')
        widget_clean_scribble.show()
        widget_filter_segmentation.show()
        widget_split_and_merge_from_scribbles.call_button.text = f'Split / Merge - < {DEFAULT_KEY_BINDING_PROOFREAD} >'
        return None

    segmentation_handler.update_scribbles_from_viewer(viewer)

    @thread_worker
    def func():
        if segmentation_handler.is_locked():
            return None

        if segmentation_handler.scribbles.sum() == 0:
            napari_formatted_logging('No scribbles found', thread='Proofreading tool')
            return None

        segmentation_handler.lock()
        new_seg, region_slice, bboxes = split_merge_from_seeds(
            segmentation_handler.scribbles,
            segmentation_handler.segmentation,
            image=image_data,
            bboxes=segmentation_handler.bboxes,
            max_label=segmentation_handler.max_label,
            correct_labels=segmentation_handler.corrected_cells,
        )

        segmentation_handler.update_after_proofreading(viewer, new_seg, region_slice, bboxes)
        segmentation_handler.unlock()

    worker = func()
    worker.start()


@magicgui(call_button='Extract correct labels')
def widget_filter_segmentation() -> Future[LayerDataTuple]:
    if not segmentation_handler.status:
        napari_formatted_logging(
            'Proofreading widget not initialized. Run the proofreading widget tool once first',
            thread='Export correct labels',
            level='error',
        )
        raise ValueError('Proofreading widget not initialized. Run the proofreading widget tool once first')

    future = Future()

    @thread_worker
    def func():
        if segmentation_handler.is_locked():
            raise ValueError('Segmentation is locked.')

        segmentation_handler.lock()
        filtered_seg = segmentation_handler.segmentation.copy()
        filtered_seg[segmentation_handler.corrected_cells_mask == 0] = 0

        new_ps_seg = PlantSegImage(filtered_seg, segmentation_handler.seg_properties)
        new_seg_layer = new_ps_seg.to_napari_layer()

        segmentation_handler.unlock()
        return new_seg_layer

    def on_done(result):
        future.set_result(result)

    worker = func()
    worker.returned.connect(on_done)
    worker.start()
    return future


widget_filter_segmentation.hide()


def setup_proofreading_keybindings(viewer):
    @viewer.bind_key(DEFAULT_KEY_BINDING_PROOFREAD)
    def _widget_split_and_merge_from_scribbles(_viewer: napari.Viewer):
        widget_split_and_merge_from_scribbles(viewer=_viewer)

    @viewer.bind_key(DEFAULT_KEY_BINDING_CLEAN)
    def _widget_clean_scribble(_viewer: napari.Viewer):
        widget_clean_scribble(viewer=_viewer)

    @viewer.mouse_double_click_callbacks.append
    def _add_label_to_corrected(_viewer: napari.Viewer, event):
        # Maybe it would be better to run this callback only if the layer is active
        # if _viewer.layers.selection.active.name == CORRECTED_CELLS_LAYER_NAME:
        if CORRECTED_CELLS_LAYER_NAME in _viewer.layers:
            widget_add_label_to_corrected(viewer=viewer, position=event.position)
