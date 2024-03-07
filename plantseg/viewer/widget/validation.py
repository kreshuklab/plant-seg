"""Widget input validation"""

from psygnal import Signal
from functools import wraps

def change_handler(*widgets, init=True, debug=False):
    def decorator_change_handler(handler):
        @wraps(handler)
        def wrapper(*args):
            source = Signal.sender()
            emitter = Signal.current_emitter()
            if debug:
                # print(f"{emitter}: {source} = {args!r}")
                print(f"EVENT '{str(emitter.name)}': {source.name:>20} = {args!r}")
                # print(f"                 {source.name:>14}.value = {source.value}")
            return handler(*args)

        for widget in widgets:
            widget.changed.connect(wrapper)
            if init:
                widget.changed(widget.value)
        return wrapper

    return decorator_change_handler


def get_image_volume_from_layer(image):
    """Used for widget parameter validation in `change_handler`s."""
    image = image.data[0] if image.multiscale else image.data
    if not all(hasattr(image, attr) for attr in ("shape", "ndim", "__getitem__")):
        image = np.asanyarray(image)
    return image


def widgets_inactive(*widgets, active):
    """Toggle visibility of widgets."""
    for widget in widgets:
        widget.visible = active


def widgets_valid(*widgets, valid):
    """Toggle background warning color of widgets."""
    for widget in widgets:
        widget.native.setStyleSheet("" if valid else "background-color: lightcoral")


