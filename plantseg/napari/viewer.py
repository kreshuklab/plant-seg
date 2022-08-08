import napari

from plantseg.napari.containers import get_gasp_workflow, get_preprocessing_workflow, get_main


def run_viewer():
    viewer = napari.Viewer()

    main_container = get_main()
    container1 = get_preprocessing_workflow()
    container2 = get_gasp_workflow()

    main_w = viewer.window.add_dock_widget(main_container, name='Main')
    container1_w = viewer.window.add_dock_widget(container1, name='Data - Processing')
    container2_w = viewer.window.add_dock_widget(container2, name='GASP Workflow')

    viewer.window._qt_window.tabifyDockWidget(main_w, container1_w)
    viewer.window._qt_window.tabifyDockWidget(main_w, container2_w)
    napari.run()



