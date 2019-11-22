import tkinter
from tkinter import filedialog
import os
import yaml
stick_all = tkinter.N + tkinter.S + tkinter.E + tkinter.W


class Files2Process:
    def __init__(self, config):
        """ Browse for file and directory """
        self.files = tkinter.StringVar()
        if config["path"] is None:
            self.files.set(os.path.expanduser("~"))
        else:
            self.files.set(config["path"])
        self.config = config

    def browse_for_file(self):
        """ browse for file """
        current_file_dir, _ = os.path.split(self.files.get())
        current_file_dir = (os.path.expanduser("~") if len(os.path.expanduser("~")) > len(current_file_dir)
                            else current_file_dir)

        file_name = filedialog.askopenfilename(initialdir=current_file_dir,
                                               title="Select file",
                                               filetypes=(("h5 files", "*.h5"),
                                                          ("hdf files", "*.hdf"),
                                                          ("tiff files", "*.tiff"),
                                                          ("tif files", "*.tif"),))
        self.files.set(file_name)
        self.config["path"] = file_name

    def browse_for_directory(self):
        """ browse for directory """
        current_file_dir, _ = os.path.split(self.files.get())
        current_file_dir = (os.path.expanduser("~") if len(os.path.expanduser("~")) > len(current_file_dir)
                            else current_file_dir)
        dire_name = filedialog.askdirectory(initialdir=current_file_dir,
                                            title="Select directory")
        self.files.set(dire_name)
        self.config["path"] = dire_name


def convert_rgb(rgb=(0, 0, 0)):
    """ rgb to tkinter friendly format"""
    rgb = tuple(rgb)
    return "#%02x%02x%02x" % rgb


def list_models():
    """ list model zoo """
    model_config = os.path.split(os.path.abspath('__file__'))[0]
    model_config = os.path.join(model_config, "models", "models_zoo.yaml")
    model_config = yaml.load(open(model_config, 'r'),
                             Loader=yaml.FullLoader)
    models = list(model_config.keys())
    return models


def var_to_tkinter(var):
    """ transform python variables in tkinter variables"""
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


class StdoutRedirect:
    """
    A class for redirecting stdout to this Text widget.
    """

    def __init__(self, widget):
        self.widget = widget

    def write(self, data):
        self.widget.configure(state='normal')
        self.widget.insert(tkinter.END, f"{data}")
        self.widget.configure(state='disabled')
        self.widget.yview_moveto(1)
        self.widget.update()

    def flush(self):
        pass


def report_error(data):
    """ creates pop up and show error messages """
    data = data if type(data) is str else f"Unknown Error. Error type: {type(data)} \n {data}"

    popup = tkinter.Tk()
    popup.title("Error")
    popup["bg"] = "white"
    tkinter.Grid.rowconfigure(popup, 0, weight=1)
    tkinter.Grid.columnconfigure(popup, 0, weight=1)

    x = tkinter.Label(popup, bg=convert_rgb((240, 192, 208)), text=data)
    x.grid(column=0, row=0, padx=10, pady=10, sticky=stick_all)
