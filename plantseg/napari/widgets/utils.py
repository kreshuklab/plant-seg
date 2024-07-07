from magicgui.widgets import Widget


def return_value_if_widget(x):
    if isinstance(x, Widget):
        return x.value
    return x
