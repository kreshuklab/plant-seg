import os
import pickle
from collections import deque
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path

import napari
import numpy as np
from magicgui import magicgui
from napari.layers import Image, Labels
from napari.qt.threading import thread_worker
from napari.types import LayerDataTuple
from napari.utils import CyclicLabelColormap

from plantseg.core.image import ImageProperties, PlantSegImage, SemanticType
from plantseg.functionals.proofreading.split_merge_tools import split_merge_from_seeds
from plantseg.functionals.proofreading.utils import get_bboxes
from plantseg.viewer_napari import log

DEFAULT_KEY_BINDING_PROOFREAD = 'n'
DEFAULT_KEY_BINDING_CLEAN = 'j'
SCRIBBLES_LAYER_NAME = 'Scribbles'
CORRECTED_CELLS_LAYER_NAME = 'Correct Labels'
MAX_UNDO_ACTIONS = 10
try:
    MAX_UNDO_ACTIONS = int(os.getenv('PLANTSEG_MAX_UNDO_ACTIONS', str(MAX_UNDO_ACTIONS)))
except ValueError:
    log('Invalid value for PLANTSEG_MAX_UNDO_ACTIONS, using default: 10', thread='Proofreading', level='warning')


def copy_if_not_none(obj):
    """Returns a copy of the object if it's not None."""
    return None if obj is None else obj.copy()


class ProofreadingHandler:
    """Handler for managing segmentation proofreading and corrections.

    This class handles the state of the segmentation, corrected cells, scribbles,
    and bounding boxes, while allowing for interactions such as undoing changes.
    """

    _status: bool
    _current_seg_layer_name: str | None
    _corrected_cells: set
    _segmentation: np.ndarray | None
    _current_seg_properties: dict | None
    _corrected_cells_mask: np.ndarray | None
    _scribbles: np.ndarray | None
    _bboxes: np.ndarray | None

    _lock: bool = False
    scale: tuple | None = None
    scribbles_layer_name = SCRIBBLES_LAYER_NAME
    corrected_cells_layer_name = CORRECTED_CELLS_LAYER_NAME
    correct_cells_cmap = CyclicLabelColormap(
        colors=[(0.76388469, 0.02003777, 0.61156412, 1.0), (0.76388469, 0.02003777, 0.61156412, 1.0)],
        name='Corrected Cells',
    )

    # Stacks for saving snapshots
    _history_undo: deque = deque(maxlen=MAX_UNDO_ACTIONS)
    _history_redo: deque = deque(maxlen=MAX_UNDO_ACTIONS)

    def __init__(self):
        """Initializes the ProofreadingHandler with an inactive state."""
        self._status = False

    @contextmanager
    def lock_manager(self):
        """Context manager for locking and unlocking proofreading handler."""
        self._lock = True
        try:
            yield
        finally:
            self._lock = False

    def is_locked(self):
        """Checks if the proofreading handler is locked."""
        return self._lock

    @property
    def status(self):
        """Returns the proofreading status."""
        return self._status

    @property
    def seg_layer_name(self):
        """Returns the current segmentation layer name."""
        return self._current_seg_layer_name

    @property
    def seg_properties(self) -> ImageProperties:
        """Returns the properties of the current segmentation."""
        return self._current_seg_properties  # TODO: fix type hint

    @property
    def segmentation(self):
        """Returns the current segmentation data."""
        return self._segmentation

    @property
    def scribbles(self):
        """Returns the current scribbles."""
        return self._scribbles

    @property
    def corrected_cells_mask(self):
        """Returns the mask for corrected cells."""
        return self._corrected_cells_mask

    @property
    def bboxes(self):
        """Returns the bounding boxes (bboxes) for the segmentation."""
        return self._bboxes

    @property
    def max_label(self):
        """Returns the maximum label value in the segmentation."""
        return self.segmentation.max()

    @property
    def corrected_cells(self):
        """Returns the set of corrected cells."""
        return self._corrected_cells

    def save_to_history(self):
        """Saves the current state to the undo history and clears the redo history."""
        self._history_undo.append(self._capture_state())
        self._history_redo.clear()  # Clear the redo stack when new actions are made

    def _capture_state(self):
        """Captures the current state of the handler."""
        return {
            'segmentation': copy_if_not_none(self._segmentation),
            'corrected_cells': copy_if_not_none(self._corrected_cells),
            'scribbles': copy_if_not_none(self._scribbles),
            'corrected_cells_mask': copy_if_not_none(self._corrected_cells_mask),
            'bboxes': copy_if_not_none(self._bboxes),
        }

    def _restore_state(self, state):
        """Restores a given state."""
        self._segmentation = state['segmentation']
        self._corrected_cells = state['corrected_cells']
        self._scribbles = state['scribbles']
        self._corrected_cells_mask = state['corrected_cells_mask']
        self._bboxes = state['bboxes']

    def _perform_undo_redo(self, history_pop, history_append, viewer, action_name):
        """Generalized function to handle undo and redo actions."""
        if not history_pop:
            log(f"No more actions to {action_name}.", thread=action_name.capitalize())
            return

        current_state = self._capture_state()
        last_state = history_pop.pop()

        history_append.append(current_state)
        self._restore_state(last_state)

        self._update_to_viewer(viewer, self._segmentation, self.seg_layer_name)
        self.update_scribble_to_viewer(viewer)
        log(f'{action_name.capitalize()} completed', thread=action_name.capitalize())

    def undo(self, viewer: napari.Viewer):
        """Restores the previous state from the history stack."""
        self._perform_undo_redo(self._history_undo, self._history_redo, viewer, 'undo')

    def redo(self, viewer: napari.Viewer):
        """Restores the next state from the redo history."""
        self._perform_undo_redo(self._history_redo, self._history_undo, viewer, 'redo')

    def setup(self, segmentation: PlantSegImage):
        """Initializes the proofreading handler with a new segmentation.

        Args:
            segmentation (PlantSegImage): The segmentation image to set up.
        """
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
        """Resets the proofreading handler to its initial state."""
        self._status = False
        self._current_seg_layer_name = None
        self._corrected_cells = set()

        self._segmentation = None
        self._corrected_cells_mask = None
        self._scribbles = None
        self._bboxes = None
        self.scale = None

    def toggle_corrected_cell(self, cell_id: int):
        """Toggles a cell as corrected or not.

        Args:
            cell_id (int): The ID of the cell to toggle.
        """
        self._toggle_corrected_cell(cell_id)
        self._update_masks(cell_id)

    def _toggle_corrected_cell(self, cell_id: int):
        """Adds or removes the cell from the corrected set.

        Args:
            cell_id (int): The ID of the cell to toggle.
        """
        if cell_id in self._corrected_cells:
            self._corrected_cells.remove(cell_id)
        else:
            self._corrected_cells.add(cell_id)

    def _update_masks(self, cell_id: int):
        """Updates the corrected cells mask with the toggled cell.

        Args:
            cell_id (int): The ID of the cell to update.
        """
        mask = self._segmentation == cell_id

        self._corrected_cells_mask[mask] += 1
        self._corrected_cells_mask = self._corrected_cells_mask % 2  # act as a toggle

    @staticmethod
    def _update_to_viewer(viewer: napari.Viewer, data: np.ndarray, layer_name: str, **kwargs):
        """Updates a layer in the viewer with new data.

        Args:
            viewer (napari.Viewer): The current Napari viewer instance.
            data (np.ndarray): The new data to update the layer with.
            layer_name (str): The name of the layer to update.
        """
        if layer_name in viewer.layers:
            viewer.layers[layer_name].data = data
            viewer.layers[layer_name].refresh()

        else:
            viewer.add_labels(data, name=layer_name, **kwargs)

    @staticmethod
    def _update_slice_to_viewer(viewer: napari.Viewer, data: np.ndarray, layer_name: str, region_slice):
        """Updates a slice of a layer in the viewer.

        Args:
            viewer (napari.Viewer): The current Napari viewer instance.
            data (np.ndarray): The new slice data to update.
            layer_name (str): The name of the layer to update.
            region_slice (tuple): The region slice to update in the layer.
        """
        if layer_name in viewer.layers:
            viewer.layers[layer_name].data[region_slice] = data
            viewer.layers[layer_name].refresh()
        else:
            raise ValueError(f'Layer {layer_name} not found in viewer')

    def update_scribble_to_viewer(self, viewer: napari.Viewer):
        """Updates the scribble layer in the viewer.

        Args:
            viewer (napari.Viewer): The current Napari viewer instance.
        """
        self._update_to_viewer(viewer, self._scribbles, self.scribbles_layer_name, scale=self.scale)

    def update_scribbles_from_viewer(self, viewer: napari.Viewer):
        """Fetches scribbles data from the viewer and updates the handler.

        Args:
            viewer (napari.Viewer): The current Napari viewer instance.
        """
        self._scribbles = viewer.layers[self.scribbles_layer_name].data

    def reset_scribbles(self):
        """Resets the scribble data to an empty state."""
        self._scribbles = np.zeros_like(self._segmentation).astype(np.uint16)

    def preserve_labels(self, viewer: napari.Viewer, layer_name: str):
        """Preserves labels on a layer in the viewer.

        Args:
            viewer (napari.Viewer): The current Napari viewer instance.
            layer_name (str): The name of the layer to preserve.
        """
        viewer.layers[layer_name].preserve_labels = True
        viewer.layers[layer_name].refresh()

    def update_corrected_cells_mask_to_viewer(self, viewer: napari.Viewer):
        """Updates the corrected cells mask in the viewer.

        Args:
            viewer (napari.Viewer): The current Napari viewer instance.
        """
        self._update_to_viewer(
            viewer,
            self.corrected_cells_mask,
            self.corrected_cells_layer_name,
            scale=self.scale,
            colormap=self.correct_cells_cmap,
            opacity=1,
        )
        self.preserve_labels(viewer, self.corrected_cells_layer_name)

    def update_corrected_cells_mask_slice_to_viewer(
        self, viewer: napari.Viewer, slice_data: np.ndarray, region_slice: tuple[slice, ...]
    ):
        """Updates a slice of the corrected cells mask in the viewer.

        Args:
            viewer (napari.Viewer): The current Napari viewer instance.
            slice_data (np.ndarray): The data to update the slice with.
            region_slice (tuple[slice, ...]): The region slice to update.
        """
        self._update_slice_to_viewer(viewer, slice_data, self.corrected_cells_layer_name, region_slice)
        self.preserve_labels(viewer, self.corrected_cells_layer_name)

    def update_after_proofreading(
        self, viewer: napari.Viewer, seg_slice: np.ndarray, region_slice: tuple[slice, ...], bbox: np.ndarray
    ):
        """Updates the viewer after proofreading is completed.

        Args:
            viewer (napari.Viewer): The current Napari viewer instance.
            seg_slice (np.ndarray): The segmentation slice to update.
            region_slice (tuple[slice, ...]): The region slice to update in the viewer.
            bbox (np.ndarray): The bounding box to update.
        """
        self._bboxes = bbox
        self._update_slice_to_viewer(viewer, seg_slice, self.seg_layer_name, region_slice)

    def reset_corrected_cells_mask(self):
        """Resets the corrected cells mask to an empty state."""
        self._corrected_cells_mask = np.zeros_like(self._segmentation).astype(np.uint16)

    def save_state_to_disk(self, filepath: Path):
        """Saves the current state to disk."""
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)  # Create directories if they don't exist
            with filepath.open('wb') as f:
                pickle.dump(self._capture_state(), f)
            log(f'State saved successfully to {filepath}', thread='Proofreading tool')
        except Exception as e:
            log(f'Error saving state to {filepath}: {e}', thread='Proofreading tool', level='error')

    def load_state_from_disk(self, filepath: Path, viewer: napari.Viewer):
        """Loads a saved state from disk."""
        if not filepath.exists():
            log(f'Error: State file not found at {filepath}', thread='Proofreading tool', level='error')
            return

        try:
            with filepath.open('rb') as f:
                state = pickle.load(f)
            self._restore_state(state)
            self._update_to_viewer(viewer, self._segmentation, self.seg_layer_name)
            self.update_scribble_to_viewer(viewer)
            log(f'State loaded successfully from {filepath}', thread='Proofreading tool')
        except Exception as e:
            log(f'Error loading state from {filepath}: {e}', thread='Proofreading tool', level='error')


segmentation_handler = ProofreadingHandler()


@magicgui(call_button=f'Clean scribbles - < {DEFAULT_KEY_BINDING_CLEAN} >')
def widget_clean_scribble(viewer: napari.Viewer):
    """Cleans the scribbles layer in the Napari viewer.

    Args:
        viewer (napari.Viewer): The current Napari viewer instance.
    """
    if not segmentation_handler.status:
        log('Proofreading widget not initialized. Run the proofreading widget tool once first', thread='Clean scribble')

    if 'Scribbles' not in viewer.layers:
        log('Scribble Layer not defined. Run the proofreading widget tool once first', thread='Clean scribble')
        return None

    segmentation_handler.reset_scribbles()
    segmentation_handler.update_scribble_to_viewer(viewer)


def widget_add_label_to_corrected(viewer: napari.Viewer, position: tuple[int, ...]):
    """Adds or removes a label at a given position to/from the corrected cells.

    Args:
        viewer (napari.Viewer): The current Napari viewer instance.
        position (tuple[int, ...]): The position of the cell in the viewer.
    """
    if segmentation_handler.corrected_cells_layer_name not in viewer.layers:
        return None

    position = [int(p / s) for p, s in zip(position, segmentation_handler.scale)]
    cell_id = segmentation_handler.segmentation[*position]
    segmentation_handler.toggle_corrected_cell(cell_id)
    segmentation_handler.update_corrected_cells_mask_to_viewer(viewer)


def initialize_proofreading(viewer: napari.Viewer, segmentation: PlantSegImage, state: Path | None = None) -> None:
    """Initializes the proofreading tool with the given segmentation.

    Args:
        viewer (napari.Viewer): The current Napari viewer instance.
        segmentation (PlantSegImage): The segmentation image to use.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    segmentation_handler.reset()
    segmentation_handler.setup(segmentation)
    segmentation_handler.update_scribble_to_viewer(viewer)
    segmentation_handler.update_corrected_cells_mask_to_viewer(viewer)
    widget_proofreading_initialisation.call_button.text = 'Re-initialize Proofreading'
    setup_proofreading_widget()
    log('Proofreading initialized', thread='Proofreading tool')
    if state:
        segmentation_handler.load_state_from_disk(state, viewer)
        log('State loaded successfully', thread='Proofreading tool')


@magicgui(
    call_button='Initialize Proofreading',
    segmentation={
        'label': 'Segmentation',
        'tooltip': 'The segmentation layer to proofread',
    },
    state={
        'label': 'Resume from file',
        'mode': 'r',
        'tooltip': 'Load a previous proofreading state from a pickle (*.pkl) file',
    },
    are_you_sure={'label': 'I understand this resets everything', 'visible': False},
)
def widget_proofreading_initialisation(
    viewer: napari.Viewer,
    segmentation: Labels,
    state: Path | None = None,
    are_you_sure: bool = False,
) -> None:
    """Initializes the proofreading widget.

    Args:
        viewer (napari.Viewer): The current Napari viewer instance.
        segmentation (Labels): The segmentation layer.
        state (Path | None): Path to a previous state file (optional).
    """
    if segmentation.name in [
        SCRIBBLES_LAYER_NAME,
        CORRECTED_CELLS_LAYER_NAME,
    ]:  # Avoid re-initializing with proofreading helper layers
        log(
            'Scribble or corrected cells layer is not intended to be proofread, choose a segmentation',
            thread='Proofreading tool',
            level='error',
        )
        return

    if segmentation_handler.status and not are_you_sure:
        log(
            'Proofreading is already initialized. Are you sure you want to reset everything?',
            thread='Proofreading tool',
            level='warning',
        )
        widget_proofreading_initialisation.are_you_sure.show()
        widget_proofreading_initialisation.call_button.text = 'I understand, please re-initialise!!'
        return

    ps_segmentation = PlantSegImage.from_napari_layer(segmentation)
    initialize_proofreading(viewer, ps_segmentation, state)
    widget_proofreading_initialisation.are_you_sure.value = False
    widget_proofreading_initialisation.are_you_sure.hide()
    widget_proofreading_initialisation.call_button.text = 'Re-initialize Proofreading'
    widget_proofreading_initialisation.segmentation.choices = [  # Avoid re-initializing with proofreading helper layers
        layer for layer in viewer.layers if layer.name not in [SCRIBBLES_LAYER_NAME, CORRECTED_CELLS_LAYER_NAME]
    ]


@magicgui(
    call_button=f'Split / Merge - < {DEFAULT_KEY_BINDING_PROOFREAD} >',
    image={
        'label': 'Boundary image',
        'tooltip': 'Probability map (prediction) or raw image of boundaries as reference',
    },
)
def widget_split_and_merge_from_scribbles(
    viewer: napari.Viewer,
    image: Image,
):
    """Splits or merges segments using scribbles as seeds for corrections.

    Args:
        viewer (napari.Viewer): The current Napari viewer instance.
        image (Image): The probability map or raw image layer.
    """
    if not segmentation_handler.status:
        log('Proofreading is not initialized. Run the initialization widget first.', thread='Proofreading tool')
        return

    ps_image = PlantSegImage.from_napari_layer(image)

    if ps_image.semantic_type == SemanticType.RAW:
        log(
            'Pmap/Image layer appears to be a raw image and not a boundary probability map. '
            'For the best proofreading results, try to use a boundaries probability layer '
            '(e.g. from the Run Prediction widget)',
            thread='Proofreading tool',
            level='warning',
        )

    if ps_image.is_multichannel:
        log(
            'Pmap/Image layer appears to be a multichannel image. '
            'Proofreading does not support multichannel images. ',
            thread='Proofreading tool',
            level='error',
        )

    segmentation_handler.update_scribbles_from_viewer(viewer)

    @thread_worker
    def func():
        if segmentation_handler.is_locked():
            return None

        if segmentation_handler.scribbles.sum() == 0:
            log('No scribbles found', thread='Proofreading tool')
            return None

        with segmentation_handler.lock_manager():
            segmentation_handler.save_to_history()

            new_seg, region_slice, bboxes = split_merge_from_seeds(
                segmentation_handler.scribbles,
                segmentation_handler.segmentation,
                image=ps_image.get_data(),
                bboxes=segmentation_handler.bboxes,
                max_label=segmentation_handler.max_label,
                correct_labels=segmentation_handler.corrected_cells,
            )

            segmentation_handler.update_after_proofreading(viewer, new_seg, region_slice, bboxes)

    worker = func()
    worker.start()


@magicgui(call_button='Freeze correct labels')
def widget_filter_segmentation() -> Future[LayerDataTuple]:
    """Extracts corrected labels from the segmentation.

    Returns:
        Future[LayerDataTuple]: A future that will return the extracted segmentation layer.
    """
    if not segmentation_handler.status:
        log(
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

        with segmentation_handler.lock_manager():
            filtered_seg = segmentation_handler.segmentation.copy()
            filtered_seg[segmentation_handler.corrected_cells_mask == 0] = 0

            properties = segmentation_handler.seg_properties

            new_seg_properties = ImageProperties(
                name=f'{properties.name}_corrected',
                semantic_type=SemanticType.LABEL,
                voxel_size=properties.voxel_size,
                image_layout=properties.image_layout,
                original_voxel_size=properties.original_voxel_size,
            )
            new_ps_seg = PlantSegImage(filtered_seg, new_seg_properties)
            new_seg_layer_tuple = new_ps_seg.to_napari_layer_tuple()

        return new_seg_layer_tuple

    def on_done(result):
        future.set_result(result)

    worker = func()
    worker.returned.connect(on_done)
    worker.start()
    return future


@magicgui(call_button='Undo Last Action')
def widget_undo(viewer: napari.Viewer):
    """Undo the last proofreading action.

    Args:
        viewer (napari.Viewer): The current Napari viewer instance.
    """
    if not segmentation_handler.status:
        log('Proofreading widget not initialized. Nothing to undo.', thread='Undo')
        return
    segmentation_handler.undo(viewer)


@magicgui(call_button='Redo Last Action')
def widget_redo(viewer: napari.Viewer):
    """Redo the last undone action.

    Args:
        viewer (napari.Viewer): The current Napari viewer instance.
    """
    if not segmentation_handler.status:
        log('Proofreading widget not initialized. Nothing to redo.', thread='Redo')
        return
    segmentation_handler.redo(viewer)


@magicgui(
    call_button='Save current proofreading snapshot',
    filepath={
        'label': 'File path',
        'mode': 'w',
    },
    images_saved={'label': 'I saved boundary references and original segmentations'},
)
def widget_save_state(filepath: Path = Path.home(), images_saved: bool = False):
    """Saves the current proofreading state to disk.

    Args:
        filepath (str): The filepath to save the state to.

    """
    if images_saved:
        segmentation_handler.save_state_to_disk(filepath)
    else:
        log('Please save the pmap and segmentation images before saving the state', thread='Save State', level='error')


def setup_proofreading_keybindings(viewer):
    """Sets up keybindings for the proofreading tool in Napari.

    Args:
        viewer (napari.Viewer): The current Napari viewer instance.
    """

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


activation_list_proofreading = [
    widget_split_and_merge_from_scribbles,
    widget_clean_scribble,
    widget_filter_segmentation,
    widget_undo,
    widget_redo,
    widget_save_state,
]

for widget in activation_list_proofreading:
    widget.hide()


def setup_proofreading_widget():
    for widget in activation_list_proofreading:
        widget.show()
