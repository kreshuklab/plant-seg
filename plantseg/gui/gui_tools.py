import tkinter
from tkinter import filedialog
import os
import yaml
from plantseg import plantseg_global_path
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
    model_config = os.path.join(plantseg_global_path, 'resources', 'models_zoo.yaml')
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

    def __init__(self, net_resolution, config, preprocessing_menu, postprocessing_menu, font=None):
        self.net_resolution = net_resolution
        self.config = config
        self.preprocessing_menu, self.postprocessing_menu = preprocessing_menu, postprocessing_menu

        popup = tkinter.Toplevel()
        popup.title("Auto Re-Scale")
        popup.configure(bg="white")
        self.popup = popup
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
        label1 = tkinter.Label(popup_instructions, bg="white", text="Please insert your data resolution",
                               font=font)
        label1.grid(column=0,
                    row=0,
                    padx=10,
                    pady=10,
                    sticky=self.stick_all)

        from plantseg.gui.gui_widgets import ListEntry
        self.list_entry = ListEntry(popup, "Resolution of the data set to predict (zxy \u03BCm): ", row=1, column=0, type=float,
                                    font=font)
        self.list_entry(net_resolution, [])

        popup_button = tkinter.Frame(popup)
        popup_button.configure(bg="white")
        tkinter.Grid.rowconfigure(popup_button, 0, weight=1)
        tkinter.Grid.columnconfigure(popup_button, 0, weight=1)
        popup_button.grid(row=2, column=0, sticky=self.stick_all)
        button = tkinter.Button(popup_button, bg="white", text="Apply", command=self.update_input_resolution,
                                font=font)
        button.grid(column=0,
                    row=0,
                    padx=10,
                    pady=10,
                    sticky=self.stick_all)

    def update_input_resolution(self):
        self.user_input = [self.list_entry.tk_value[i].get() for i in range(3)]
        scaling_factor = [self.user_input[i] / self.net_resolution[i] for i in range(3)]

        [self.preprocessing_menu.custom_key["factor"].tk_value[i].set(scaling_factor[i])
         for i in range(3)]
        [self.postprocessing_menu.post_pred_obj.custom_key["factor"].tk_value[i].set(1.0 / scaling_factor[i])
         for i in range(3)]
        [self.postprocessing_menu.post_seg_obj.custom_key["factor"].tk_value[i].set(1.0 / scaling_factor[i])
         for i in range(3)]

        self.config = self.preprocessing_menu.check_and_update_config(self.config,
                                                                      dict_key="preprocessing")

        self.config = self.postprocessing_menu.post_pred_obj.check_and_update_config(self.config,
                                                                                     dict_key="cnn_postprocessing")

        self.config = self.postprocessing_menu.post_pred_obj.check_and_update_config(self.config,
                                                                                     dict_key="segmentation"
                                                                                              "_postprocessing")
        self.popup.destroy()
