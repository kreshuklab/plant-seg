from pathlib import Path
from typing import Tuple
from magicgui import magicgui
from plantseg.models.zoo import model_zoo
from plantseg.viewer.logging import napari_formatted_logging
from plantseg.viewer.widget.predictions import (
    widget_iterative_unet_predictions,
    widget_test_all_unet_predictions,
    widget_unet_predictions)
from plantseg.viewer.widget.segmentation import widget_lifted_multicut, widget_fix_over_under_segmentation_from_nuclei


@magicgui(call_button='Add Custom Model',
          new_model_name={'label': 'New model name'},
          model_location={'label': 'Model location',
                          'mode': 'd'},
          resolution={'label': 'Resolution', 'options': {'step': 0.00001}},
          description={'label': 'Description'},
          dimensionality={'label': 'Dimensionality',
                          'tooltip': 'Dimensionality of the model (2D or 3D). '
                                     'Any 2D model can be used for 3D data.',
                          'widget_type': 'ComboBox',
                          'choices': model_zoo.get_unique_dimensionalities()},
          modality={'label': 'Microscopy modality',
                    'tooltip': 'Modality of the model (e.g. confocal, light-sheet ...).',
                    'widget_type': 'ComboBox',
                    'choices': model_zoo.get_unique_modalities()},
          output_type={'label': 'Prediction type',
                       'widget_type': 'ComboBox',
                       'tooltip': 'Type of prediction (e.g. cell boundaries predictions or nuclei...).',
                       'choices': model_zoo.get_unique_output_types()},

          )
def widget_add_custom_model(new_model_name: str = 'custom_model',
                            model_location: Path = Path.home(),
                            resolution: Tuple[float, float,
                                              float] = (1., 1., 1.),
                            description: str = 'New custom model',
                            dimensionality: str = model_zoo.get_unique_dimensionalities()[
                                0],
                            modality: str = model_zoo.get_unique_modalities()[
                                0],
                            output_type: str = model_zoo.get_unique_output_types()[0]) -> None:
    finished, error_msg = model_zoo.add_custom_model(new_model_name=new_model_name,
                                                     location=model_location,
                                                     resolution=resolution,
                                                     description=description,
                                                     dimensionality=dimensionality,
                                                     modality=modality,
                                                     output_type=output_type)

    if finished:
        napari_formatted_logging(f'New model {new_model_name} added to the list of available models.',
                                 level='info',
                                 thread='Add Custom Model')
        widget_unet_predictions.model_name.choices = model_zoo.list_models()
    else:
        napari_formatted_logging(f'Error adding new model {new_model_name} to the list of available models: '
                                 f'{error_msg}',
                                 level='error',
                                 thread='Add Custom Model')


registered_extra_widgets = {"Lifted MultiCut": widget_lifted_multicut,
                            "Add Custom Model": widget_add_custom_model,
                            "Test all UNet": widget_test_all_unet_predictions,
                            "Iterative UNet": widget_iterative_unet_predictions,
                            "Proofread from Nuclei": widget_fix_over_under_segmentation_from_nuclei}


@magicgui(auto_call=True,
          widget_name={'label': 'Widget Selection',
                       'tooltip': 'Select the widget to show.',
                       'choices': list(registered_extra_widgets.keys())})
def widget_extra_manager(widget_name: str) -> None:
    napari_formatted_logging(f'Showing {widget_name} widget',
                             thread='Extra Widget Manager',
                             level='info')
    for key, value in registered_extra_widgets.items():
        if key == widget_name:
            value.show()
        else:
            value.hide()


for _widget in registered_extra_widgets.values():
    _widget.hide()
