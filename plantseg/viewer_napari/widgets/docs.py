import webbrowser
from pathlib import Path

from magicgui import magicgui
from magicgui.widgets import Container, create_widget

from plantseg.viewer_napari import log

LOGO_PATH = (
    Path(__file__).resolve().parent.parent.parent / "resources" / "logo_white.png"
)
DOCS_URL = "https://kreshuklab.github.io/plant-seg/"


def create_doc_container() -> Container:
    """Creates a container with a documentation button and a logo."""
    container = Container(
        widgets=[
            create_widget(
                widget_type="PushButton",
                label="Open Documentation",
            ),
        ],
        label=f'<img src="{LOGO_PATH}">',
        layout="horizontal",
        labels=False,
    )
    return container


@magicgui(auto_call=True)
def widget_docs():
    """MagicGUI widget to hold documentation components."""
    return


doc_container_widget = create_doc_container()
widget_docs.insert(0, doc_container_widget)


@doc_container_widget.changed.connect
def open_docs(event):
    """Open the documentation URL in the default web browser when the button is clicked."""
    webbrowser.open(DOCS_URL)
    log(
        message=f"Docs webpage opened: {DOCS_URL}", thread="Documentation", level="info"
    )
