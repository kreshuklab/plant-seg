import webbrowser

from PyQt5.QtCore import Qt
from magicgui.widgets import MainWindow

from plantseg.napari.widgets import widget_open_file, widget_export_stacks, widget_show_info, widget_infos
from plantseg.napari.widgets import widget_gaussian_smoothing, widget_rescaling
from plantseg.napari.widgets import widget_unet_predictions, widget_dt_ws, widget_agglomeration

STYLE_SLIDER = "font-size: 9pt;"


def setup_menu(container, path=None):
    def _callback():
        if path is not None:
            webbrowser.open(path)

    container.create_menu_item(menu_name="Help", item_name="Open Documentation", callback=_callback)

    container._widget._layout.setAlignment(Qt.AlignTop)
    return container


def get_data_io():
    container = MainWindow(
        widgets=[widget_open_file, widget_export_stacks, widget_show_info, widget_infos], labels=False
    )
    container = setup_menu(
        container,
        path="https://kreshuklab.github.io/plant-seg/chapters/plantseg_interactive_napari/import_export/",
    )
    return container


def get_preprocessing_tab():
    # widget_cropping.crop_z.native.setStyleSheet(STYLE_SLIDER) # TODO remove comment when widget_cropping is implemented
    container = MainWindow(
        widgets=[
            widget_gaussian_smoothing,
            widget_rescaling,
        ],
        labels=False,
    )
    container = setup_menu(
        container,
        path='https://kreshuklab.github.io/plant-seg/chapters/plantseg_interactive_napari/data_processing/',
    )
    return container


def get_main_tab():
    container = MainWindow(widgets=[widget_unet_predictions, widget_dt_ws, widget_agglomeration], labels=False)
    container = setup_menu(
        container,
        path='https://kreshuklab.github.io/plant-seg/chapters/plantseg_interactive_napari/unet_gasp_workflow/',
    )
    return container


def get_extras_tab():
    # widet_fix_over_under_segmentation_from_nuclei.threshold.native.setStyleSheet(STYLE_SLIDER)
    # widget_fix_over_under_segmentation_from_nuclei.quantile.native.setStyleSheet(STYLE_SLIDER)
    container = MainWindow(
        widgets=[],
        labels=False,
    )
    container = setup_menu(
        container, path='https://kreshuklab.github.io/plant-seg/chapters/plantseg_interactive_napari/extra/'
    )
    return container


def get_proofreading_tab():
    container = MainWindow(
        widgets=[],
        labels=False,
    )
    container = setup_menu(
        container, path='https://kreshuklab.github.io/plant-seg/chapters/plantseg_interactive_napari/proofreading/'
    )
    return container
