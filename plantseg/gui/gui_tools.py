import os
import tkinter
from tkinter import filedialog

import yaml

from plantseg import plantseg_global_path
from plantseg.gui import stick_all, stick_ew, var_to_tkinter, convert_rgb

current_model = None

######################################################################################################################
#
# Menu entries Prototypes
#
######################################################################################################################
class MenuEntry:
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


class SimpleEntry:
    """ Standard open entry widget """
    def __init__(self, frame, text="Text", row=0, column=0, _type=str, _font=None):
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
                    sticky=tkinter.E)

        obj_collection.append(label1)
        obj_collection.append(entry1)
        return obj_collection


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
        print(current_model)
        path_model_config = self.get_model_path()

        model_config = yaml.load(open(path_model_config, 'r'),
                                 Loader=yaml.FullLoader)

        net_resolution = model_config[current_model]["resolution"]
        AutoResPopup(net_resolution, current_model, self.tk_value, self.font)

    @staticmethod
    def get_model_path():
        # Working directory path + relative dir structure to yaml file
        config_path = os.path.join(plantseg_global_path, "resources", "models_zoo.yaml")
        return config_path


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

######################################################################################################################
#
# Generic GUI tools
#
######################################################################################################################

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

    def fileno(self):
        return 0


def report_error(data, font=None):
    """ creates pop up and show error messages """
    data = data if type(data) is str else f"Unknown Error. Error type: {type(data)} \n {data}"

    default = "The complete error message is reported in the terminal." \
              " Please, if the error persist let us know by opening an issue on https://github.com/hci-unihd/plant-seg."

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
        self.net_resolution = net_resolution
        popup = tkinter.Toplevel()
        popup.title("Auto Re-Scale")
        popup.configure(bg="white")
        self.popup = popup

        # Place popup
        tkinter.Grid.rowconfigure(popup, 0, weight=2)
        tkinter.Grid.rowconfigure(popup, 1, weight=1)
        tkinter.Grid.rowconfigure(popup, 2, weight=1)

        tkinter.Grid.columnconfigure(popup, 0, weight=1)
        self.stick_all = tkinter.N + tkinter.S + tkinter.W + tkinter.E

        popup_instructions = tkinter.Frame(popup)
        tkinter.Grid.rowconfigure(popup_instructions, 0, weight=1)
        tkinter.Grid.columnconfigure(popup_instructions, 0, weight=1)
        popup_instructions.grid(row=0, column=0, sticky=self.stick_all)
        popup_instructions.configure(bg="white")

        text0 = f"The model you currently selected is {net_name}"
        text1 = f"The model was trained with data at voxel resolution of {self.net_resolution} (zxy \u03BCm)"
        text2 = f"It is generally useful to rescale your input data to match the resolution of the original data"

        label0 = tkinter.Label(popup_instructions, bg="white", text=text0,
                               font=font)
        label0.grid(column=0,
                    row=0,
                    padx=10,
                    pady=10,
                    sticky=self.stick_all)

        label1 = tkinter.Label(popup_instructions, bg="white", text=text1,
                               font=font)
        label1.grid(column=0,
                    row=1,
                    padx=10,
                    pady=10,
                    sticky=self.stick_all)
        label2 = tkinter.Label(popup_instructions, bg="white", text=text2,
                               font=font)
        label2.grid(column=0,
                    row=2,
                    padx=10,
                    pady=10,
                    sticky=self.stick_all)

        self.list_entry = ListEntry(popup, "Input your data resolution (zxy \u03BCm): ", row=1, column=0, type=float,
                                    font=font)
        self.list_entry(net_resolution, [])
        self.scale_factor = []
        self.tk_value = tk_value

        popup_button = tkinter.Frame(popup)
        popup_button.configure(bg="white")
        tkinter.Grid.rowconfigure(popup_button, 0, weight=1)
        tkinter.Grid.columnconfigure(popup_button, 0, weight=1)
        popup_button.grid(row=2, column=0, sticky=self.stick_all)
        button = tkinter.Button(popup_button, bg="white", text="Apply Rescaling Factor",
                                command=self.update_input_resolution,
                                font=font)
        button.grid(column=0,
                    row=0,
                    padx=10,
                    pady=10,
                    sticky=self.stick_all)
        text = "N.B. Rescaling input data to the training data resolution has shown to be very helpful in some cases" \
               " and detrimental in others."
        label = tkinter.Label(popup_button, bg="white", text=text,
                               font=font)
        label.grid(column=0,
                    row=1,
                    padx=10,
                    pady=10,
                    sticky=self.stick_all)

    def update_input_resolution(self):
        self.user_input = [self.list_entry.tk_value[i].get() for i in range(3)]
        self.scale_factor = [self.user_input[i] / self.net_resolution[i] for i in range(3)]
        [self.tk_value[i].set(float(self.scale_factor[i])) for i in range(3)]
        self.popup.destroy()
