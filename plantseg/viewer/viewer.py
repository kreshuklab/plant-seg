import napari
from napari.utils.notifications import show_info

from plantseg.viewer.containers import get_extra_seg, get_extra_pred
from plantseg.viewer.containers import get_gasp_workflow, get_preprocessing_workflow, get_main
from plantseg.viewer.widget.proofreading.proofreading import widget_split_and_merge_from_scribbles, widget_clean_scribble


def run_viewer():
    viewer = napari.Viewer()
    main_container = get_main()
    main_w = viewer.window.add_dock_widget(main_container, name='Main')

    viewer.bind_key('p', widget_split_and_merge_from_scribbles)
    viewer.bind_key('c', widget_clean_scribble)

    for _containers, name in [(get_preprocessing_workflow(), 'Data - Processing'),
                              (get_gasp_workflow(), 'UNet + Segmentation'),
                              (get_extra_pred(), 'Extra-Pred'),
                              (get_extra_seg(), 'Extra-Seg'),
                              ]:
        _container_w = viewer.window.add_dock_widget(_containers, name=name)
        viewer.window._qt_window.tabifyDockWidget(main_w, _container_w)

    show_info('Napari - Plantseg is ready!')
    napari.run()
