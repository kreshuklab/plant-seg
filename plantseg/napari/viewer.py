import napari
from plantseg.napari.containers import get_gasp_workflow, get_preprocessing_workflow, get_main
from plantseg.napari.containers import get_extra_seg, get_extra_pred


def run_viewer():
    viewer = napari.Viewer()
    main_container = get_main()
    main_w = viewer.window.add_dock_widget(main_container, name='Main')

    for _containers, name in [(get_preprocessing_workflow(), 'Data - Processing'),
                              (get_gasp_workflow(), 'UNet + GASP Workflow'),
                              (get_extra_pred(), 'Extra-Pred'),
                              (get_extra_seg(), 'Extra-Seg')]:
        _container_w = viewer.window.add_dock_widget(_containers, name=name)
        viewer.window._qt_window.tabifyDockWidget(main_w, _container_w)

    napari.run()




