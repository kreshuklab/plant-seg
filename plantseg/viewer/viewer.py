import napari
import numpy as np
from napari.utils.notifications import show_info

from plantseg.viewer.containers import get_extra_seg, get_extra_pred
from plantseg.viewer.containers import get_gasp_workflow, get_preprocessing_workflow, get_main
from plantseg.viewer.widget.proofreading.proofreading import default_key_binding_clean, default_key_binding_split_merge
from plantseg.viewer.widget.proofreading.proofreading import widget_clean_scribble
from plantseg.viewer.widget.proofreading.proofreading import widget_split_and_merge_from_scribbles
from napari.layers import Labels


def run_viewer():
    viewer = napari.Viewer()
    main_container = get_main()
    main_w = viewer.window.add_dock_widget(main_container, name='Main')

    @viewer.bind_key(default_key_binding_split_merge)
    def _widget_split_and_merge_from_scribbles(viewer):
        widget_split_and_merge_from_scribbles(viewer=viewer)

    @viewer.bind_key(default_key_binding_clean)
    def _widget_clean_scribble(viewer):
        widget_clean_scribble(viewer=viewer)

    @viewer.mouse_drag_callbacks.append
    def callback(_viewer, event):
        pos = event.position
        from plantseg.viewer.widget.proofreading.proofreading import widget_add_label_to_corrected
        widget_add_label_to_corrected(viewer=_viewer, position=pos)


    for _containers, name in [(get_preprocessing_workflow(), 'Data - Processing'),
                              (get_gasp_workflow(), 'UNet + Segmentation'),
                              (get_extra_pred(), 'Extra-Pred'),
                              (get_extra_seg(), 'Extra-Seg'),
                              ]:
        _container_w = viewer.window.add_dock_widget(_containers, name=name)
        viewer.window._qt_window.tabifyDockWidget(main_w, _container_w)

    show_info('Napari - Plantseg is ready!')
    napari.run()
