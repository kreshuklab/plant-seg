from concurrent.futures import Future

from magicgui import magicgui
from napari.types import LayerDataTuple
from pathlib import Path
from typing import List, Union
from plantseg.io import smart_load
from napari.layers import Image, Layer


@magicgui(
    call_button="Open file",
    path={"label": "Pick a file to open (tiff or h5)"},
    stack_type={
        "label": "Data type",
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        "choices": ['image', 'labels']},
    key={"label": "key (h5 only)"},
    channel={"label": "channel (tiff only)"}

)
def open_file(path: Path = Path.home(),
              stack_type: str = 'image',
              open_all: bool = True,
              key: str = None,
              channel: int = 0
              ) -> LayerDataTuple:
    raw, info = smart_load(path, key=key)
    return raw, {}, stack_type


@magicgui(
    call_button="Export stack",
    path={"label": "Pick a file to open (tiff or h5)"},
)
def export_stack(images: List[Layer],
                 path: Path = Path.home(),
           ) -> None:
    pass
