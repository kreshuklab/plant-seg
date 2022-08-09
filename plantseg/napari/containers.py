from magicgui.widgets import Container

from plantseg.napari.widget.dataprocessing import widget_generic_preprocessing, widget_cropping, widget_add_layers
from plantseg.napari.widget.io import open_file, export_stacks
from plantseg.napari.widget.predictions import widget_unet_predictions
from plantseg.napari.widget.segmentation import widget_dt_ws, widget_gasp
from plantseg.napari.widget.segmentation import widget_multicut, widget_lifted_multicut


def get_main():
    container = Container(widgets=[open_file,
                                   export_stacks],
                          labels=False)

    return container


def get_preprocessing_workflow():
    container = Container(widgets=[widget_generic_preprocessing,
                                   widget_cropping,
                                   widget_add_layers],
                          labels=False)
    return container


def get_gasp_workflow():
    container = Container(widgets=[widget_unet_predictions,
                                   widget_dt_ws,
                                   widget_gasp],
                          labels=False)
    return container


def get_extra():
    container = Container(widgets=[widget_multicut,
                                   widget_lifted_multicut,
                                   ],
                          labels=False)
    return container
