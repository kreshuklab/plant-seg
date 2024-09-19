from concurrent.futures import Future
from pathlib import Path

from magicgui import magicgui
from napari.layers import Image, Labels

from plantseg.core.image import PlantSegImage
from plantseg.datasets.dataset_handler import Dataset, delete_dataset, list_datasets, load_dataset

########################################################################################################################
#                                                                                                                      #
# Dataset Management Widget                                                                                             #
#                                                                                                                      #
########################################################################################################################


@magicgui(
    call_button='Create New Dataset',
    name={
        'label': 'Dataset Name',
        'tooltip': 'Name of the dataset to create.',
    },
    directory={
        'label': 'Directory',
        'tooltip': 'Directory where the dataset will be created.',
        'mode': 'd',
    },
)
def widget_dataset_create(
    name: str = "dataset",
    directory: Path = Path(".").absolute(),
) -> None:
    Dataset(name=name, directory=directory)


@magicgui(
    call_button='Add Image',
    name={
        'label': 'Dataset Name',
        'tooltip': 'Name of the dataset to load.',
        'choices': list_datasets(),
    },
)
def widget_dataset_add_image(
    name: str = list_datasets()[0],
    raw: Image = None,
    label: Labels = None,
    mask: Labels = None,
) -> None:
    dataset = load_dataset(name)

    if raw is None:
        raise ValueError("Raw image must be provided.")

    if label is None:
        raise ValueError("Label image must be provided.")

    ps_raw = PlantSegImage.from_napari_layer(raw)
    ps_label = PlantSegImage.from_napari_layer(label)
    if mask is not None:
        ps_mask = PlantSegImage.from_napari_layer(mask)

    else:
        ps_mask = None

    dataset.add_image(raw=ps_raw, label=ps_label, mask=ps_mask)


@magicgui(
    call_button='Delete Dataset',
    name={
        'label': 'Dataset Name',
        'tooltip': 'Name of the dataset to delete.',
        'choices': list_datasets(),
    },
)
def widget_dataset_delete(
    name: str = list_datasets()[0],
) -> None:
    delete_dataset(name)


@magicgui(
    call_button='Start Fine-Tuning',
)
def start_fine_tuning(
    name: str = list_datasets()[0],
):
    dataset = load_dataset(name)

    #
    # Fine-tuning code here
    #
