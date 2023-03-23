from typing import Union

import napari
import numpy as np
from magicgui import magicgui
from napari.layers import Labels, Image
from napari.qt.threading import thread_worker
from plantseg.viewer.logging import formatted_logging

from plantseg.viewer.widget.proofreading.utils import get_bboxes
from plantseg.viewer.widget.proofreading.split_merge_tools import split_merge_from_seeds

default_key_binding_split_merge = 'n'
default_key_binding_clean = 'b'
SCRIBBLES_LAYER_NAME = 'Scribbles'
CORRECTED_CELLS_LAYER_NAME = 'Correct Labels'


class ProofreadingHandler:
    _status: bool
    _current_seg_layer_name: Union[str, None]
    _corrected_cells: set
    _segmentation: Union[np.ndarray, None]
    _corrected_cells_mask: Union[np.ndarray, None]
    _scribbles: Union[np.ndarray, None]
    _bboxes: Union[np.ndarray, None]

    _lock: bool = False
    scale: Union[tuple, None] = None
    scribbles_layer_name = SCRIBBLES_LAYER_NAME
    corrected_cells_layer_name = CORRECTED_CELLS_LAYER_NAME
    correct_cells_cmap = {0: None,
                          1: (0.76388469, 0.02003777, 0.61156412, 1.)
                          }

    def __init__(self):
        self._status = False

    @property
    def status(self):
        return self._status

    @property
    def seg_layer_name(self):
        return self._current_seg_layer_name

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

    def setup(self, segmentation_layer: Labels):
        # make sure all fields are reset
        self.reset()

        self._status = True
        segmentation = segmentation_layer.data
        self._current_seg_layer_name = segmentation_layer.name
        self.scale = segmentation_layer.scale

        self._segmentation = segmentation
        self.reset_scribbles()
        self.reset_corrected_cells_mask()

        self._bboxes = get_bboxes(segmentation)

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

    def update_corrected_cells_mask_to_viewer(self, viewer: napari.Viewer):
        self._update_to_viewer(viewer,
                               self.corrected_cells_mask,
                               self.corrected_cells_layer_name,
                               scale=self.scale,
                               color=self.correct_cells_cmap)

    def update_corrected_cells_mask_slice_to_viewer(self, viewer: napari.Viewer,
                                                    slice_data: np.ndarray,
                                                    region_slice: tuple[slice, ...]):
        self._update_slice_to_viewer(viewer, slice_data, self.corrected_cells_layer_name, region_slice)

    def update_after_proofreading(self, viewer: napari.Viewer,
                                  seg_slice: np.ndarray,
                                  region_slice: tuple[slice, ...],
                                  bbox: np.ndarray):

        self._bboxes = bbox
        self._update_slice_to_viewer(viewer, seg_slice, self.seg_layer_name, region_slice)

    def reset_corrected_cells_mask(self):
        self._corrected_cells_mask = np.zeros_like(self._segmentation).astype(np.uint16)


segmentation_handler = ProofreadingHandler()


@magicgui(call_button=f'Clean scribbles - < {default_key_binding_clean} >')
def widget_clean_scribble(viewer: napari.Viewer):
    if 'Scribbles' not in viewer.layers:

        formatted_logging('Scribble Layer not defined. Run the proofreading widget tool once first',
                          thread='clean scribble')
        return None

    segmentation_handler.reset_scribbles()
    segmentation_handler.update_scribble_to_viewer(viewer)


def widget_add_label_to_corrected(viewer: napari.Viewer, position: tuple[int, ...]):
    if segmentation_handler.corrected_cells_layer_name not in viewer.layers:
        return None

    if len(position) == 2:
        position = [0, *position]

    position = [int(p / s) for p, s in zip(position, segmentation_handler.scale)]
    cell_id = segmentation_handler.segmentation[position[0], position[1], position[2]]
    segmentation_handler.toggle_corrected_cell(cell_id)
    segmentation_handler.update_corrected_cells_mask_to_viewer(viewer)


def initialize_proofreading(viewer: napari.Viewer, segmentation_layer: Labels) -> bool:
    if segmentation_handler.scribbles_layer_name not in viewer.layers:
        segmentation_handler.reset()

    if segmentation_handler.corrected_cells_layer_name not in viewer.layers:
        segmentation_handler.reset()

    if segmentation_handler.seg_layer_name != segmentation_layer.name:
        segmentation_handler.reset()

    if segmentation_handler.status:
        return False

    segmentation_handler.setup(segmentation_layer)
    segmentation_handler.update_scribble_to_viewer(viewer)
    segmentation_handler.update_corrected_cells_mask_to_viewer(viewer)
    return True


@magicgui(call_button=f'Split/Merge from scribbles - < {default_key_binding_split_merge} >',
          segmentation={'label': 'Segmentation'},
          image={'label': 'Image'})
def widget_split_and_merge_from_scribbles(viewer: napari.Viewer,
                                          segmentation: Labels,
                                          image: Image) -> None:
    if initialize_proofreading(viewer, segmentation):
        formatted_logging('Proofreading initialized', thread='proofreading tool')
        return None

    segmentation_handler.update_scribbles_from_viewer(viewer)

    @thread_worker
    def func():
        if segmentation_handler.is_locked():
            return None

        if segmentation_handler.scribbles.sum() == 0:
            formatted_logging('No scribbles found', thread='proofreading tool')
            return None

        segmentation_handler.lock()
        new_seg, region_slice, bboxes = split_merge_from_seeds(segmentation_handler.scribbles,
                                                               segmentation_handler.segmentation,
                                                               image=image.data,
                                                               bboxes=segmentation_handler.bboxes,
                                                               max_label=segmentation_handler.max_label,
                                                               correct_labels=segmentation_handler.corrected_cells)

        segmentation_handler.update_after_proofreading(viewer, new_seg, region_slice, bboxes)
        segmentation_handler.unlock()

    worker = func()
    worker.start()
