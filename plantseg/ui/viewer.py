import napari

from plantseg.ui.containers import get_extra
from plantseg.ui.containers import get_gasp_workflow, get_preprocessing_workflow, get_main
from plantseg.ui.containers import get_dataset_workflow
from plantseg.ui.logging import napari_formatted_logging
from plantseg.ui.widgets.proofreading.proofreading import setup_proofreading_keybindings


def run_viewer():
    viewer = napari.Viewer()
    main_container = get_main()
    main_w = viewer.window.add_dock_widget(main_container, name='Main')

    setup_proofreading_keybindings(viewer)

    for _containers, name in [(get_preprocessing_workflow(), 'Data - Processing'),
                              (get_gasp_workflow(), 'UNet + Segmentation'),
                              (get_dataset_workflow(), 'Datasets Management'),
                              (get_extra(), 'Extra-Workflows'),
                              ]:
        _container_w = viewer.window.add_dock_widget(_containers, name=name)
        viewer.window._qt_window.tabifyDockWidget(main_w, _container_w)

    napari_formatted_logging('Plantseg is ready!', thread='Main', level='info')
    napari.run()
