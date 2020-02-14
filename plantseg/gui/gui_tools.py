import glob
import os
import tkinter
from shutil import copy2, rmtree
from tkinter import filedialog

import yaml

from plantseg import custom_zoo, home_path, PLANTSEG_MODELS_DIR, model_zoo_path
from plantseg.__version__ import __version__
from plantseg.gui import stick_all, stick_ew, var_to_tkinter, convert_rgb, PLANTSEG_GREEN
from plantseg.pipeline import gui_logger
from plantseg.pipeline.steps import H5_EXTENSIONS, TIFF_EXTENSIONS
from plantseg.pipeline.utils import read_tiff_voxel_size, read_h5_voxel_size

current_model = None
current_segmentation = None


######################################################################################################################
#
# Menu entries Prototypes
#
######################################################################################################################


class SimpleEntry:
    """ Standard open entry widget """

    def __init__(self, frame, text="Text", large_bar=False, row=0, column=0, _type=str, _font=None):
        self.frame = tkinter.Frame(frame)

        self.text = f"{text}"

        self.type = _type
        self.style = {"bg": "white",
                      "padx": 10,
                      "pady": 10,
                      "row_weights": [1],
                      "columns_weights": [1, 1],
                      "height": 4,
                      }

        self.frame["bg"] = self.style["bg"]
        self.font = _font

        [tkinter.Grid.rowconfigure(self.frame, i, weight=w) for i, w in enumerate(self.style["row_weights"])]
        [tkinter.Grid.columnconfigure(self.frame, i, weight=w) for i, w in enumerate(self.style["columns_weights"])]
        self.frame.grid(row=row, column=column, sticky=stick_ew)

        self.tk_value = None

        if large_bar:
            self.stick = stick_ew
        else:
            self.stick = tkinter.E

    def __call__(self, value, obj_collection):
        self.tk_value = var_to_tkinter(self.type(value))

        label1 = tkinter.Label(self.frame, bg=self.style["bg"], text=self.text, anchor="w", font=self.font)
        label1.grid(column=0,
                    row=0,
                    padx=self.style["padx"],
                    pady=self.style["pady"],
                    sticky=tkinter.W)

        entry1 = tkinter.Entry(self.frame, textvar=self.tk_value, font=self.font)
        entry1.grid(column=1,
                    row=0,
                    padx=self.style["padx"],
                    pady=self.style["pady"],
                    sticky=self.stick)

        obj_collection.append(label1)
        obj_collection.append(entry1)
        return obj_collection


class SliderEntry:
    """ Standard open entry widget """

    def __init__(self,
                 frame, text="Text", row=0, column=0, data_range=(0, 1, 0.1),
                 is_not_in_dtws=False, _type=float, _font=None):
        self.frame = tkinter.Frame(frame)

        self.text = f"{text}"
        self.min, self.max, self.interval = data_range

        self.type = _type
        self.style = {"bg": "white",
                      "padx": 10,
                      "pady": 10,
                      "row_weights": [1],
                      "columns_weights": [1, 1],
                      "height": 4,
                      }

        self.frame["bg"] = self.style["bg"]
        self.font = _font

        [tkinter.Grid.rowconfigure(self.frame, i, weight=w) for i, w in enumerate(self.style["row_weights"])]
        [tkinter.Grid.columnconfigure(self.frame, i, weight=w) for i, w in enumerate(self.style["columns_weights"])]
        self.frame.grid(row=row, column=column, sticky=stick_ew)

        self.tk_value = None

    def __call__(self, value, obj_collection):
        global current_segmentation
        label1 = tkinter.Label(self.frame, bg=self.style["bg"], text=self.text, anchor="w", font=self.font)
        label1.grid(column=0,
                    row=0,
                    padx=self.style["padx"],
                    pady=self.style["pady"],
                    sticky=tkinter.W)

        entry1 = tkinter.Scale(self.frame, from_=self.min, to=self.max, resolution=self.interval,
                               orient=tkinter.HORIZONTAL, font=self.font)
        entry1.configure(bg="white")
        entry1.configure(troughcolor=convert_rgb(PLANTSEG_GREEN))
        entry1.configure(length=200)
        entry1.set(self.type(value))

        entry1.grid(column=1,
                    row=0,
                    padx=self.style["padx"],
                    pady=self.style["pady"],
                    sticky=tkinter.E)

        self.tk_value = entry1

        obj_collection.append(label1)
        obj_collection.append(entry1)
        return obj_collection


class MenuEntry:
    """ Standard menu widget """

    def __init__(self, frame, text="Text", row=0, column=0, menu=(),
                 is_model=False, is_segmentation=False, default=None, font=None):
        self.frame = tkinter.Frame(frame)

        self.text = f"{text}"

        self.menu = menu
        self.style = {"bg": "white",
                      "padx": 10,
                      "pady": 10,
                      "row_weights": [1],
                      "columns_weights": [1, 1],
                      "height": 4,
                      }

        self.frame["bg"] = self.style["bg"]
        self.font = font

        [tkinter.Grid.rowconfigure(self.frame, i, weight=w) for i, w in enumerate(self.style["row_weights"])]
        [tkinter.Grid.columnconfigure(self.frame, i, weight=w) for i, w in enumerate(self.style["columns_weights"])]
        self.frame.grid(row=row, column=column, sticky=stick_ew)

        self.tk_value = tkinter.StringVar()
        if default is None:
            self.tk_value.set(sorted(list(self.menu))[0])
        else:
            if type(default) == bool:
                default = "True" if default else "False"
            self.tk_value.set(default)

        self.is_model = is_model
        if self.is_model:
            self.update_model_name(default)

        self.is_segmentation = is_segmentation
        if self.is_segmentation:
            self.update_segmentation_name(default)

    def __call__(self, value, obj_collection):

        label1 = tkinter.Label(self.frame, bg=self.style["bg"], text=self.text, anchor="w", font=self.font)
        label1.grid(column=0,
                    row=0,
                    padx=self.style["padx"],
                    pady=self.style["pady"],
                    sticky=stick_all)

        if self.is_model:
            entry1 = tkinter.OptionMenu(self.frame, self.tk_value, *self.menu, command=self.update_model_name)

        elif self.is_segmentation:
            entry1 = tkinter.OptionMenu(self.frame, self.tk_value, *self.menu, command=self.update_segmentation_name)

        else:
            entry1 = tkinter.OptionMenu(self.frame, self.tk_value, *self.menu)

        entry1.config(font=self.font)
        entry1["menu"].config(bg="white")
        entry1.config(bg="white")
        entry1.grid(column=1,
                    row=0,
                    padx=self.style["padx"],
                    pady=self.style["pady"],
                    sticky=stick_all)

        obj_collection.append(label1)
        obj_collection.append(entry1)
        return obj_collection

    def update_model_name(self, value):
        global current_model
        current_model = value

    def update_segmentation_name(self, value):
        global current_segmentation
        current_segmentation = value


class BoolEntry:
    """ Standard boolean widget """

    def __init__(self, frame, text="Text", row=0, column=0, font=None):
        self.frame = tkinter.Frame(frame)

        self.text = f"{text}"

        self.style = {"bg": "white",
                      "padx": 10,
                      "pady": 10,
                      "row_weights": [1],
                      "columns_weights": [1, 1],
                      "height": 4,
                      }

        self.frame["bg"] = self.style["bg"]
        self.font = font

        [tkinter.Grid.rowconfigure(self.frame, i, weight=w) for i, w in enumerate(self.style["row_weights"])]
        [tkinter.Grid.columnconfigure(self.frame, i, weight=w) for i, w in enumerate(self.style["columns_weights"])]
        self.frame.grid(row=row, column=column, sticky=stick_ew)

        self.tk_value = None

    def __call__(self, value, obj_collection):
        self.tk_value = tkinter.BooleanVar(value)

        label1 = tkinter.Label(self.frame, bg=self.style["bg"], text=self.text, anchor="w", font=self.font)
        label1.grid(column=0,
                    row=0,
                    padx=self.style["padx"],
                    pady=self.style["pady"],
                    sticky=stick_ew)

        entry1 = tkinter.Checkbutton(self.frame, variable=self.tk_value, bg=self.style["bg"], font=self.font)
        entry1.grid(column=1,
                    row=0,
                    padx=self.style["padx"],
                    pady=self.style["pady"],
                    sticky=stick_ew)

        obj_collection.append(label1)
        obj_collection.append(entry1)
        return obj_collection


class FilterEntry:
    """ Special widget for filter """

    def __init__(self, frame, text="Text", row=0, column=0, font=None):
        self.frame = tkinter.Frame(frame)

        self.text = f"{text}"

        self.style = {"bg": "white",
                      "padx": 10,
                      "pady": 10,
                      "row_weights": [1],
                      "columns_weights": [1, 1, 3, 1],
                      "height": 4,
                      }

        self.frame["bg"] = self.style["bg"]
        self.font = font

        [tkinter.Grid.rowconfigure(self.frame, i, weight=w) for i, w in enumerate(self.style["row_weights"])]
        [tkinter.Grid.columnconfigure(self.frame, i, weight=w) for i, w in enumerate(self.style["columns_weights"])]
        self.frame.grid(row=row, column=column, sticky=stick_ew)

        self.tk_value = None

    def __call__(self, value, obj_collection):
        self.tk_value = [tkinter.BooleanVar(), tkinter.StringVar(), tkinter.DoubleVar()]
        self.tk_value[0].set(False)
        self.tk_value[1].set("gaussian")
        self.tk_value[2].set(1.0)

        label1 = tkinter.Label(self.frame, bg=self.style["bg"], text=self.text, anchor="w", font=self.font)
        label1.grid(column=0,
                    row=0,
                    padx=self.style["padx"],
                    pady=self.style["pady"],
                    sticky=stick_all)

        entry1 = tkinter.Checkbutton(self.frame, variable=self.tk_value[0], bg=self.style["bg"], font=self.font)
        entry1.grid(column=1,
                    row=0,
                    padx=self.style["padx"],
                    pady=self.style["pady"],
                    sticky=stick_all)

        entry2 = tkinter.OptionMenu(self.frame, self.tk_value[1], *{"median", "gaussian"})
        entry2["menu"].config(font=self.font)
        entry2["menu"].config(bg="white")
        entry2.config(font=self.font)
        entry2.config(bg="white")
        entry2.grid(column=2,
                    row=0,
                    padx=self.style["padx"],
                    pady=self.style["pady"],
                    sticky=stick_all)

        entry3 = tkinter.Entry(self.frame, textvar=self.tk_value[2], width=3)
        entry3.grid(column=3,
                    row=0,
                    padx=self.style["padx"],
                    pady=self.style["pady"],
                    sticky=stick_all)

        obj_collection.append(label1)
        obj_collection.append(entry1)
        obj_collection.append(entry2)
        obj_collection.append(entry3)
        return obj_collection


class MenuEntryStride:
    """ Standard menu widget """

    def __init__(self, frame, text="Text", row=0, column=0, menu=(), is_model=False, default=None, font=None):
        self.frame = tkinter.Frame(frame)

        self.text = f"{text}"

        self.menu = menu
        self.style = {"bg": "white",
                      "padx": 10,
                      "pady": 10,
                      "row_weights": [1],
                      "columns_weights": [1, 1],
                      "height": 4,
                      }

        self.frame["bg"] = self.style["bg"]
        self.font = font

        [tkinter.Grid.rowconfigure(self.frame, i, weight=w) for i, w in enumerate(self.style["row_weights"])]
        [tkinter.Grid.columnconfigure(self.frame, i, weight=w) for i, w in enumerate(self.style["columns_weights"])]
        self.frame.grid(row=row, column=column, sticky=stick_ew)

        self.tk_value = tkinter.StringVar()
        if default is None:
            self.tk_value.set(sorted(list(self.menu))[0])
        else:
            if type(default) == bool:
                default = "True" if default else "False"
            self.tk_value.set(default)

        self.is_model = is_model
        if self.is_model:
            self.update_model_name(default)

    def __call__(self, value, obj_collection):

        label1 = tkinter.Label(self.frame, bg=self.style["bg"], text=self.text, anchor="w", font=self.font)
        label1.grid(column=0,
                    row=0,
                    padx=self.style["padx"],
                    pady=self.style["pady"],
                    sticky=stick_all)

        if self.is_model:
            entry1 = tkinter.OptionMenu(self.frame, self.tk_value, *self.menu, command=self.update_model_name)
        else:
            entry1 = tkinter.OptionMenu(self.frame, self.tk_value, *self.menu)

        entry1.config(font=self.font)
        entry1["menu"].config(bg="white")
        entry1.config(bg="white")
        entry1.grid(column=1,
                    row=0,
                    padx=self.style["padx"],
                    pady=self.style["pady"],
                    sticky=stick_all)

        obj_collection.append(label1)
        obj_collection.append(entry1)
        return obj_collection

    def update_model_name(self, value):
        global current_model
        current_model = value


class RescaleEntry:
    """ Special widget for rescale """

    def __init__(self, frame, text="Text", row=0, column=0, type=float, font=None):
        self.frame = tkinter.Frame(frame)

        self.text = f"{text}"
        self.type = type
        self.style = {"bg": "white",
                      "padx": 10,
                      "pady": 10,
                      "row_weights": [1],
                      "columns_weights": [1, 1, 1, 1, 1],
                      "height": 4,
                      }

        self.frame["bg"] = self.style["bg"]
        self.font = font

        [tkinter.Grid.rowconfigure(self.frame, i, weight=w) for i, w in enumerate(self.style["row_weights"])]
        [tkinter.Grid.columnconfigure(self.frame, i, weight=w) for i, w in enumerate(self.style["columns_weights"])]
        self.frame.grid(row=row, column=column, sticky=stick_ew)

        self.tk_value = None

    def __call__(self, value, obj_collection):
        tk_type = tkinter.DoubleVar if self.type is float else tkinter.IntVar

        self.tk_value = [tk_type() for _ in range(3)]
        [self.tk_value[i].set(self.type(value[i])) for i in range(3)]

        label1 = tkinter.Label(self.frame, bg=self.style["bg"], text=self.text, anchor="w", font=self.font)
        label1.grid(column=0,
                    row=0,
                    padx=self.style["padx"],
                    pady=self.style["pady"],
                    sticky=stick_ew)

        entry0 = tkinter.Button(self.frame, text="Guided", command=self.auto_rescale, bg=self.style["bg"],
                                font=self.font)
        entry0.grid(column=1,
                    row=0,
                    padx=(self.style["padx"], 0),
                    pady=self.style["pady"],
                    sticky=stick_ew)
        entry0["bg"] = "white"

        entry1 = tkinter.Entry(self.frame, textvar=self.tk_value[0], width=3, font=self.font)
        entry1.grid(column=2,
                    row=0,
                    padx=(self.style["padx"], 0),
                    pady=self.style["pady"],
                    sticky=stick_ew)
        entry1["bg"] = "white"

        entry2 = tkinter.Entry(self.frame, textvar=self.tk_value[1], width=3, font=self.font)
        entry2.grid(column=3,
                    row=0,
                    pady=self.style["pady"],
                    sticky=stick_ew)
        entry2["bg"] = "white"

        entry3 = tkinter.Entry(self.frame, textvar=self.tk_value[2], width=3, font=self.font)
        entry3.grid(column=4,
                    row=0,
                    padx=(0, self.style["padx"]),
                    pady=self.style["pady"],
                    sticky=stick_ew)
        entry3["bg"] = "white"

        obj_collection.append(label1)
        obj_collection.append(entry0)
        obj_collection.append(entry1)
        obj_collection.append(entry2)
        obj_collection.append(entry3)
        return obj_collection

    def auto_rescale(self):
        """ This method open a popup windows that automatically set the scaling
         factor from the resolution given by the user"""
        global current_model

        model_config = yaml.load(open(model_zoo_path, 'r'),
                                 Loader=yaml.FullLoader)

        net_resolution = model_config[current_model]["resolution"]
        AutoResPopup(net_resolution, current_model, self.tk_value, self.font)


class ListEntry:
    """ Standard triplet list widget """

    def __init__(self, frame, text="Text", row=0, column=0, type=float, font=None):
        self.frame = tkinter.Frame(frame)

        self.text = f"{text}"
        self.type = type
        self.style = {"bg": "white",
                      "padx": 10,
                      "pady": 10,
                      "row_weights": [1],
                      "columns_weights": [3, 1, 1, 1],
                      "height": 4,
                      }

        self.frame["bg"] = self.style["bg"]
        self.font = font

        [tkinter.Grid.rowconfigure(self.frame, i, weight=w) for i, w in enumerate(self.style["row_weights"])]
        [tkinter.Grid.columnconfigure(self.frame, i, weight=w) for i, w in enumerate(self.style["columns_weights"])]
        self.frame.grid(row=row, column=column, sticky=stick_ew)

        self.tk_value = None

    def __call__(self, value, obj_collection):
        tk_type = tkinter.DoubleVar if self.type is float else tkinter.IntVar

        self.tk_value = [tk_type() for _ in range(3)]
        [self.tk_value[i].set(self.type(value[i])) for i in range(3)]

        label1 = tkinter.Label(self.frame, bg=self.style["bg"], text=self.text, anchor="w", font=self.font)
        label1.grid(column=0,
                    row=0,
                    padx=self.style["padx"],
                    pady=self.style["pady"],
                    sticky=stick_ew)

        entry1 = tkinter.Entry(self.frame, textvar=self.tk_value[0], width=3, font=self.font)
        entry1.grid(column=1,
                    row=0,
                    padx=(self.style["padx"], 0),
                    pady=self.style["pady"],
                    sticky=stick_ew)
        entry1["bg"] = "white"

        entry2 = tkinter.Entry(self.frame, textvar=self.tk_value[1], width=3, font=self.font)
        entry2.grid(column=2,
                    row=0,
                    pady=self.style["pady"],
                    sticky=stick_ew)
        entry2["bg"] = "white"

        entry3 = tkinter.Entry(self.frame, textvar=self.tk_value[2], width=3, font=self.font)
        entry3.grid(column=3,
                    row=0,
                    padx=(0, self.style["padx"]),
                    pady=self.style["pady"],
                    sticky=stick_ew)
        entry3["bg"] = "white"

        obj_collection.append(label1)
        obj_collection.append(entry1)
        obj_collection.append(entry2)
        obj_collection.append(entry3)
        return obj_collection


class Files2Process:
    def __init__(self, config):
        """ Browse for file and directory """
        self.files = tkinter.StringVar()
        if config["path"] is None:
            self.files.set(home_path)
        else:
            self.files.set(config["path"])
        self.config = config

    def browse_for_file(self):
        """ browse for file """
        current_file_dir, _ = os.path.split(self.files.get())
        current_file_dir = (home_path if len(home_path) > len(current_file_dir)
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
        current_file_dir = (home_path if len(home_path) > len(current_file_dir)
                            else current_file_dir)
        dire_name = filedialog.askdirectory(initialdir=current_file_dir,
                                            title="Select directory")
        self.files.set(dire_name)
        self.config["path"] = dire_name


######################################################################################################################
#
# Generic GUI tools
#
######################################################################################################################
def report_error(data, font=None):
    """ creates pop up and show error messages """
    data = data if type(data) is str else f"Unknown Error. Error type: {type(data)} \n {data}"

    default = "The complete error message is reported in the terminal." \
              " If the error persists, please let us know by opening an issue on" \
              " https://github.com/hci-unihd/plant-seg."

    popup = tkinter.Tk()
    popup.title("Error")
    popup["bg"] = "white"
    tkinter.Grid.rowconfigure(popup, 0, weight=1)
    tkinter.Grid.rowconfigure(popup, 1, weight=2)
    tkinter.Grid.columnconfigure(popup, 0, weight=1)

    x = tkinter.Label(popup, bg="white", text=default, font=font)
    x.grid(column=0, row=0, padx=10, pady=10, sticky=stick_all)

    x = tkinter.Label(popup, bg=convert_rgb((240, 192, 208)), text=data, font=font)
    x.grid(column=0, row=1, padx=10, pady=10, sticky=stick_all)


class AutoResPopup:
    """ Pop up wizard for rescaling input data"""

    def __init__(self, net_resolution, net_name, tk_value, font=None):

        popup = tkinter.Toplevel()
        popup.title("Guided Re-Scale")
        popup.configure(bg="white")

        self.popup = popup
        self.net_resolution = net_resolution
        self.font = font
        self.tk_value = tk_value
        self.user_input = None
        self.scale_factor = None
        self.list_entry = None
        self.net_name = net_name

        # Place popup
        tkinter.Grid.rowconfigure(self.popup, 0, weight=2)
        tkinter.Grid.rowconfigure(self.popup, 1, weight=1)
        tkinter.Grid.rowconfigure(self.popup, 2, weight=1)

        tkinter.Grid.columnconfigure(self.popup, 0, weight=1)

        self.instructions()
        self.rescale_button_widget()

    def instructions(self):
        popup_instructions = tkinter.Frame(self.popup)
        tkinter.Grid.rowconfigure(popup_instructions, 0, weight=1)
        tkinter.Grid.columnconfigure(popup_instructions, 0, weight=1)
        popup_instructions.grid(row=0, column=0, sticky=stick_all)
        popup_instructions.configure(bg="white")

        all_text = [f"The model you currently selected is {self.net_name}",
                    f"The model was trained with data at voxel resolution of {self.net_resolution} (zxy \u03BCm)",
                    f"It is generally useful to rescale your input data to match the resolution of the original data"]

        labels = [tkinter.Label(popup_instructions, bg="white", text=text, font=self.font) for text in all_text]
        [label.grid(column=0,
                    row=i,
                    padx=10,
                    pady=10,
                    sticky=stick_all) for i, label in enumerate(labels)]

    def update_input_resolution(self):
        self.user_input = [self.list_entry.tk_value[i].get() for i in range(3)]
        self.scale_factor = [self.user_input[i] / self.net_resolution[i] for i in range(3)]
        [self.tk_value[i].set(float(self.scale_factor[i])) for i in range(3)]
        self.popup.destroy()

    def rescale_button_widget(self):
        self.list_entry = ListEntry(self.popup, "Input your data resolution (zxy \u03BCm): ",
                                    row=1, column=0, type=float,
                                    font=self.font)
        self.list_entry(self.net_resolution, [])
        self.scale_factor = []

        popup_button = tkinter.Frame(self.popup)
        popup_button.configure(bg="white")
        tkinter.Grid.rowconfigure(popup_button, 0, weight=1)
        tkinter.Grid.columnconfigure(popup_button, 0, weight=1)
        popup_button.grid(row=2, column=0, sticky=stick_all)
        button = tkinter.Button(popup_button, bg=convert_rgb(PLANTSEG_GREEN), text="Compute Rescaling Factor",
                                command=self.update_input_resolution,
                                font=self.font)

        button.grid(column=0,
                    row=0,
                    padx=10,
                    pady=10,
                    sticky=stick_all)

        search_button_frame = tkinter.Frame(self.popup)
        search_button_frame.configure(bg="white")
        tkinter.Grid.rowconfigure(search_button_frame, 0, weight=1)
        tkinter.Grid.columnconfigure(search_button_frame, 0, weight=1)
        search_button_frame.grid(row=3, column=0, sticky=stick_all)

        text = "or import resolution from file"
        label0 = tkinter.Label(search_button_frame, bg="white", text=text, font=self.font)
        label0.grid(column=0, row=0, padx=10, pady=10, sticky=stick_all)

        search_button = tkinter.Button(search_button_frame, bg=convert_rgb(PLANTSEG_GREEN),
                                       text="Compute Rescaling Factor From File",
                                       command=self.read_from_file,
                                       font=self.font)

        search_button.grid(column=0,
                           row=1,
                           padx=10,
                           pady=10,
                           sticky=stick_all)

        text = "N.B. Rescaling input data to the training data resolution has shown to be very helpful in some cases" \
               " and detrimental in others."
        label1 = tkinter.Label(search_button_frame, bg="white", text=text, font=self.font)
        label1.grid(column=0, row=2, padx=10, pady=10, sticky=stick_all)

    def read_from_file(self):
        file_dialog = Files2Process({"path": None})
        file_dialog.browse_for_file()
        path = file_dialog.config["path"]
        _, ext = os.path.splitext(path)
        if ext in H5_EXTENSIONS:
            file_resolution = read_h5_voxel_size(file_dialog.config["path"])
        elif ext in TIFF_EXTENSIONS:
            file_resolution = read_tiff_voxel_size(file_dialog.config["path"])
        else:
            raise NotImplementedError

        self.list_entry(file_resolution, [])
        self.update_input_resolution()


class LoadModelPopup:
    """ Pop up wizard for loading a neural network model"""

    def __init__(self, restart, font):
        popup = tkinter.Toplevel()
        popup.title("Load Custom Model")
        popup.configure(bg="white")

        self.popup = popup
        self.restart = restart
        self.font = font
        self.simple_entry1, self.simple_entry2, self.list_entry = None, None, None

        # Place popup
        tkinter.Grid.rowconfigure(self.popup, 0, weight=1)
        tkinter.Grid.rowconfigure(self.popup, 1, weight=2)
        tkinter.Grid.columnconfigure(self.popup, 0, weight=1)

        self.instructions()
        self.load_model_button()

    def instructions(self):
        popup_instructions = tkinter.Frame(self.popup)
        tkinter.Grid.rowconfigure(popup_instructions, 0, weight=1)
        tkinter.Grid.columnconfigure(popup_instructions, 0, weight=1)
        popup_instructions.grid(row=0, column=0, sticky=stick_all)
        popup_instructions.configure(bg="white")

        all_text = [f"In order to load a custom model you need to create a directory with the following three files: ",
                    "- Configuration file used for training (name must be config_train.yml)",
                    "- Best networks parameters (name must be best_checkpoint.pytorch)",
                    "- Last networks parameters (name must be last_checkpoint.pytorch)",
                    "All mentioned files are created when training using https://github.com/wolny/pytorch-3dunet,",
                    "Please check our repository pytorch-3dunet for training your own data.",
                    120 * " "]

        labels = [tkinter.Label(popup_instructions, bg="white", text=text, font=self.font) for text in all_text]
        [label.grid(column=0,
                    row=i,
                    padx=10,
                    pady=10,
                    sticky=stick_all) for i, label in enumerate(labels)]

    def load_model_button(self):
        popup_load = tkinter.Frame(self.popup)

        tkinter.Grid.rowconfigure(popup_load, 0, weight=1)
        tkinter.Grid.rowconfigure(popup_load, 1, weight=3)
        tkinter.Grid.rowconfigure(popup_load, 2, weight=1)

        tkinter.Grid.columnconfigure(popup_load, 0, weight=1)
        popup_load.grid(row=1, column=0, sticky=stick_all)
        popup_load.configure(bg="white")

        self.model_path = self.file_dialog_frame(popup_load, row=0, column=0)

        self.simple_entry1 = SimpleEntry(popup_load, "Model Name: ",
                                         large_bar=True,
                                         row=1, column=0, _type=str, _font=self.font)
        self.simple_entry1("custom_net", [])

        self.list_entry = ListEntry(popup_load, "Input your training data resolution (zxy \u03BCm): ",
                                    row=2, column=0, type=float,
                                    font=self.font)
        self.list_entry([1., 1., 1.], [])

        self.simple_entry2 = SimpleEntry(popup_load, "Description: ",
                                         large_bar=True,
                                         row=3, column=0, _type=str, _font=self.font)
        self.simple_entry2("", [])

        button = tkinter.Button(popup_load, bg=convert_rgb(PLANTSEG_GREEN),
                                text="Add Model Directory (This will restart PlantSeg,"
                                     " all changes not saved will be deleted)",
                                command=self.load_model,
                                font=self.font)

        button.grid(column=0,
                    row=4,
                    padx=10,
                    pady=10,
                    sticky=stick_all)

    def file_dialog_frame(self, popup, row=0, column=0):
        popup_file = tkinter.Frame(popup)
        tkinter.Grid.rowconfigure(popup_file, 0, weight=1)
        tkinter.Grid.columnconfigure(popup_file, 0, weight=1)
        tkinter.Grid.columnconfigure(popup_file, 1, weight=150)
        tkinter.Grid.columnconfigure(popup_file, 2, weight=1)

        popup_file.grid(row=row, column=column, sticky=stick_all)
        popup_file.configure(bg="white")

        self.file_dialog = Files2Process({"path": None})

        x = tkinter.Label(popup_file, bg="white", text="Custom Model Path: ", anchor="w", font=self.font)
        x.grid(column=0, row=0, padx=10, pady=10, sticky=stick_ew)

        x = tkinter.Entry(popup_file, textvar=self.file_dialog.files, font=self.font)
        x.grid(column=1, row=0, padx=0, pady=0, sticky=stick_ew)

        x = tkinter.Button(popup_file, bg="white", text="Directory",
                           command=self.file_dialog.browse_for_directory, font=self.font)
        x.grid(column=2, row=0, padx=10, pady=0, sticky=stick_ew)

    def load_model(self):
        # Model path
        path = self.file_dialog.files.get()
        # Get name
        model_name = str(self.simple_entry1.tk_value.get())
        # Get resolution
        resolution = [float(value.get()) for value in self.list_entry.tk_value]
        # Get description
        desctiption = str(self.simple_entry2.tk_value.get())

        dest_dir = os.path.join(home_path, PLANTSEG_MODELS_DIR, model_name)
        os.makedirs(dest_dir, exist_ok=True)
        all_files = glob.glob(os.path.join(path, "*"))
        all_expected_files = ['config_train.yml',
                              'last_checkpoint.pytorch',
                              'best_checkpoint.pytorch']
        for file in all_files:
            if os.path.basename(file) in all_expected_files:
                copy2(file, dest_dir)
                all_expected_files.remove(os.path.basename(file))

        if len(all_expected_files) != 0:
            msg = f'It was not possible to find in the directory specified {all_expected_files}, ' \
                  f'the model can not be loaded.'
            gui_logger.error(msg)
            self.popup.destroy()
            raise RuntimeError(msg)

        custom_zoo_dict = yaml.load(open(custom_zoo, 'r'), Loader=yaml.FullLoader)
        if custom_zoo_dict is None:
            custom_zoo_dict = {}

        custom_zoo_dict[model_name] = {}
        custom_zoo_dict[model_name]["path"] = path
        custom_zoo_dict[model_name]["resolution"] = resolution
        custom_zoo_dict[model_name]["description"] = desctiption

        with open(custom_zoo, 'w') as f:
            yaml.dump(custom_zoo_dict, f)

        gui_logger.info("Model successfully added!")
        self.restart()


class RemovePopup:
    """ Pop up wizard for removing a neural network model"""

    def __init__(self, restart, font):
        popup = tkinter.Toplevel()
        popup.title("Remove Custom Model")
        popup.configure(bg="white")

        self.popup = popup
        self.restart = restart
        self.font = font
        self.simple_entry1, self.simple_entry2, self.list_entry = None, None, None
        self.file_to_remove = tkinter.StringVar("")

        # Place popup
        tkinter.Grid.rowconfigure(self.popup, 0, weight=1)
        tkinter.Grid.columnconfigure(self.popup, 0, weight=1)

        popup_file = tkinter.Frame(self.popup)

        tkinter.Grid.rowconfigure(popup_file, 0, weight=1)
        tkinter.Grid.columnconfigure(popup_file, 0, weight=1)

        popup_file.grid(row=0, column=0, sticky=stick_all)

        popup_file.configure(bg="white")
        self.remove_model()

    def remove_model(self, row=0, column=0):
        popup_file = tkinter.Frame(self.popup)

        tkinter.Grid.rowconfigure(popup_file, 0, weight=1)
        tkinter.Grid.rowconfigure(popup_file, 1, weight=1)
        tkinter.Grid.columnconfigure(popup_file, 0, weight=1)
        tkinter.Grid.columnconfigure(popup_file, 1, weight=1)

        popup_file.grid(row=row, column=column, sticky=stick_all)
        popup_file.configure(bg="white")

        x = tkinter.Label(popup_file, bg="white", text="Type the custom model name you want to delete: ",
                          anchor="w", font=self.font)
        x.grid(column=0, row=0, padx=10, pady=10, sticky=stick_ew)

        x = tkinter.Entry(popup_file, textvar=self.file_to_remove, font=self.font)
        x.grid(column=1, row=0, padx=10, pady=10, sticky=stick_ew)

        x = tkinter.Button(popup_file, bg="white", text="Remove", command=self.delete_model, font=self.font)
        x.grid(column=1, row=1, padx=10, pady=10, sticky=stick_ew)

    def delete_model(self):
        # Delete entry in zoo custom
        self.file_to_remove = self.file_to_remove.get()
        custom_zoo_dict = yaml.load(open(custom_zoo, 'r'), Loader=yaml.FullLoader)
        if custom_zoo_dict is None:
            custom_zoo_dict = {}

        if self.file_to_remove in custom_zoo_dict:
            del custom_zoo_dict[self.file_to_remove]
        else:
            msg = f"Model {self.file_to_remove} not found." \
                  f" Please check if the name you typed is a custom model. Pre-loaded models can not be deleted."
            gui_logger.error(msg)
            self.popup.destroy()
            raise RuntimeError(msg)

        with open(custom_zoo, 'w') as f:
            yaml.dump(custom_zoo_dict, f)

        file_directory = os.path.join(home_path,
                                      PLANTSEG_MODELS_DIR,
                                      self.file_to_remove)

        if os.path.exists(file_directory):
            rmtree(file_directory)
        else:
            msg = f"Model {self.file_to_remove} not found." \
                  f" Please check if the name you typed is a custom model. Pre-loaded models can not be deleted."
            gui_logger.error(msg)
            self.popup.destroy()
            raise RuntimeError(msg)

        gui_logger.info("Model successfully removed! The effect will be visible after restarting PlantSeg")
        self.popup.destroy()


def version_popup():
    popup = tkinter.Toplevel()
    popup.title("Plantseg Version")
    tkinter.Grid.rowconfigure(popup, 0, weight=1)
    tkinter.Grid.columnconfigure(popup, 0, weight=1)
    popup.configure(bg="white")

    popup_frame = tkinter.Frame(popup)
    tkinter.Grid.rowconfigure(popup_frame, 0, weight=1)
    tkinter.Grid.columnconfigure(popup_frame, 0, weight=1)
    popup_frame.configure(bg="white")
    popup_frame.grid(row=0, column=0, sticky=stick_all)

    text = f"PlantSeg version: {__version__}\n \n" \
           f"Visit our sorce page for more info \n \n" \
           f"https://github.com/hci-unihd/plant-seg"

    label = tkinter.Label(popup_frame, bg="white", text=text)
    label.grid(column=0,
               row=0,
               padx=10,
               pady=10,
               sticky=stick_all)
