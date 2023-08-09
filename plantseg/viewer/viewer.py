import napari

from plantseg.viewer.containers import get_extra_seg, get_extra_pred
from plantseg.viewer.containers import get_gasp_workflow, get_preprocessing_workflow, get_main
from plantseg.viewer.containers import get_dataset_workflow
from plantseg.viewer.logging import napari_formatted_logging
from plantseg.viewer.widget.proofreading.proofreading import setup_proofreading_keybindings


def run_viewer():
    viewer = napari.Viewer()
    main_container = get_main()
    main_w = viewer.window.add_dock_widget(main_container, name='Main')

    setup_proofreading_keybindings(viewer)

    for _containers, name in [(get_preprocessing_workflow(), 'Data - Processing'),
                              (get_gasp_workflow(), 'UNet + Segmentation'),
                              (get_dataset_workflow(), 'Dataset'),
                              (get_extra_pred(), 'Extra-Pred'),
                              (get_extra_seg(), 'Extra-Seg'),
                              ]:
        _container_w = viewer.window.add_dock_widget(_containers, name=name)
        viewer.window._qt_window.tabifyDockWidget(main_w, _container_w)

    napari_formatted_logging('Plantseg is ready!', thread='Main', level='info')
    napari.run()
