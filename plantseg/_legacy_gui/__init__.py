import tkinter

stick_all = tkinter.N + tkinter.S + tkinter.E + tkinter.W
stick_ew = tkinter.E + tkinter.W
stick_new = tkinter.N + tkinter.E + tkinter.W
PLANTSEG_GREEN = (208, 240, 192)


def var_to_tkinter(var):
    """Transform python variables in tkinter variables"""
    if isinstance(var, bool):
        tk_var = tkinter.BooleanVar()

    elif isinstance(var, str):
        tk_var = tkinter.StringVar()

    elif isinstance(var, int):
        tk_var = tkinter.IntVar()

    elif isinstance(var, float):
        tk_var = tkinter.DoubleVar()

    elif isinstance(var, list):
        tk_var = tkinter.StringVar()

    tk_var.set(var)
    return tk_var


def convert_rgb(rgb=(0, 0, 0)):
    """RGB to tkinter friendly format"""
    rgb = tuple(rgb)
    return "#%02x%02x%02x" % rgb
