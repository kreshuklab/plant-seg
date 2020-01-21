import tkinter
from .gui_tools import convert_rgb, var_to_tkinter, list_models

stick_all = tkinter.N + tkinter.S + tkinter.E + tkinter.W
stick_n = tkinter.E + tkinter.W #tkinter.N + tkinter.S + tkinter.E + tkinter.W
stick_n2 = tkinter.N + tkinter.E + tkinter.W


class MenuEntry:
    """ Standard menu widget """
    def __init__(self, frame, text="Text", row=0, column=0, menu=(), default=None, font=None):
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
        self.frame.grid(row=row, column=column, sticky=stick_n)

        self.tk_value = tkinter.StringVar()
        if default is None:
            self.tk_value.set(sorted(list(self.menu))[0])
        else:
            if type(default) == bool:
                default = "True" if default else "False"
            self.tk_value.set(default)

    def __call__(self, value, obj_collection):

        label1 = tkinter.Label(self.frame, bg=self.style["bg"], text=self.text, anchor="w", font=self.font)
        label1.grid(column=0,
                    row=0,
                    padx=self.style["padx"],
                    pady=self.style["pady"],
                    sticky=stick_all)

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
        self.frame.grid(row=row, column=column, sticky=stick_n)

        self.tk_value = None

    def __call__(self, value, obj_collection):
        self.tk_value = var_to_tkinter(self.type(value))

        label1 = tkinter.Label(self.frame, bg=self.style["bg"], text=self.text, anchor="w", font=self.font)
        label1.grid(column=0,
                    row=0,
                    padx=self.style["padx"],
                    pady=self.style["pady"],
                    sticky=stick_n)

        entry1 = tkinter.Entry(self.frame, textvar=self.tk_value, font=self.font)
        entry1.grid(column=1,
                    row=0,
                    padx=self.style["padx"],
                    pady=self.style["pady"],
                    sticky=stick_n)

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
        self.frame.grid(row=row, column=column, sticky=stick_n)

        self.tk_value = None

    def __call__(self, value, obj_collection):
        self.tk_value = tkinter.BooleanVar(value)

        label1 = tkinter.Label(self.frame, bg=self.style["bg"], text=self.text, anchor="w", font=self.font)
        label1.grid(column=0,
                    row=0,
                    padx=self.style["padx"],
                    pady=self.style["pady"],
                    sticky=stick_n)

        entry1 = tkinter.Checkbutton(self.frame, variable=self.tk_value, bg=self.style["bg"], font=self.font)
        entry1.grid(column=1,
                    row=0,
                    padx=self.style["padx"],
                    pady=self.style["pady"],
                    sticky=stick_n)

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
        self.frame.grid(row=row, column=column, sticky=stick_n)

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
        self.frame.grid(row=row, column=column, sticky=stick_n)

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
                    sticky=stick_n)

        entry1 = tkinter.Entry(self.frame, textvar=self.tk_value[0], width=3, font=self.font)
        entry1.grid(column=1,
                    row=0,
                    padx=(self.style["padx"], 0),
                    pady=self.style["pady"],
                    sticky=stick_n)
        entry1["bg"] = "white"

        entry2 = tkinter.Entry(self.frame, textvar=self.tk_value[1], width=3, font=self.font)
        entry2.grid(column=2,
                    row=0,
                    pady=self.style["pady"],
                    sticky=stick_n)
        entry2["bg"] = "white"

        entry3 = tkinter.Entry(self.frame, textvar=self.tk_value[2], width=3, font=self.font)
        entry3.grid(column=3,
                    row=0,
                    padx=(0, self.style["padx"]),
                    pady=self.style["pady"],
                    sticky=stick_n)
        entry3["bg"] = "white"

        obj_collection.append(label1)
        obj_collection.append(entry1)
        obj_collection.append(entry2)
        obj_collection.append(entry3)
        return obj_collection


class ModuleFramePrototype:
    """
    Prototype for the main keys field.
     Every process is in the pipeline is represented by a single instance of it.
     """
    def __init__(self, frame, module_name="processing", font=None):
        self.frame = frame
        self.checkbox = None
        self.custom_key = {}
        self.obj_collection = []
        self.style = {"padx": 10, "pady": 10}
        self.show = None
        self.font = font

        self.place_module(module_name=module_name)

    def place_module(self, module_name):
        self.checkbox = tkinter.Checkbutton(self.frame, bg=convert_rgb((208, 240, 192)),
                                            text=module_name, font=self.font)
        self.checkbox.grid(column=0,
                           row=0,
                           padx=self.style["padx"],
                           pady=self.style["pady"],
                           sticky=stick_all)

    def _show_options(self, config, module):
        if self.show.get():
            self.checkbox["bg"] = convert_rgb((208, 240, 192))
            for i, (key, value) in enumerate(config[module].items()):
                if key in self.custom_key:
                    self.obj_collection = self.custom_key[key](value, self.obj_collection)

        else:
            self.checkbox["bg"] = "white"
            self.update_config(config[module])

            for obj in self.obj_collection:
                obj.grid_forget()

        return config

    def check_and_update_config(self, config, dict1, dict2):
        if self.show.get():
            if dict2:
                config[dict1][dict2]["state"] = True
                self.update_config(config[dict1][dict2])
            else:
                config[dict1]["state"] = True
                self.update_config(config[dict1])
        else:
            if dict2:
                config[dict1][dict2]["state"] = False
            else:
                config[dict1]["state"] = False

        return config

    def update_config(self, config):
        for key, obj in self.custom_key.items():
            if key in config:
                if isinstance(obj, MenuEntry):
                    str_value = obj.tk_value.get()
                    str_value = True if str_value == "True" else str_value
                    str_value = False if str_value == "False" else str_value
                    config[key] = str_value

                elif isinstance(obj, SimpleEntry):
                    str_value = obj.tk_value.get()
                    config[key] = obj.type(str_value)

                elif isinstance(obj, FilterEntry):
                    if obj.tk_value[0].get():
                        config["filter"]["state"] = True
                        config["filter"]["type"] = obj.tk_value[1].get()
                        config["filter"]["param"] = float(obj.tk_value[2].get())
                    else:
                        config["filter"]["state"] = False

                elif isinstance(obj, ListEntry):
                    values = [obj.tk_value[0].get(), obj.tk_value[1].get(), obj.tk_value[2].get()]
                    values = [obj.type(values[0]), obj.type(values[1]), obj.type(values[2])]
                    config[key] = values

    def show_options(self):
        self.config = self._show_options(self.config, self.module)


class PreprocessingFrame(ModuleFramePrototype):
    def __init__(self, frame, config, col=0, module_name="preprocessing", font=None):
        self.preprocessing_frame = tkinter.Frame(frame)
        self.preprocessing_style = {"bg": "white",
                      "padx": 10,
                      "pady": 10,
                      "row_weights": [2, 1, 1, 1, 1],
                      "columns_weights": [1],
                      "height": 4,
                      }

        self.preprocessing_frame["bg"] = self.preprocessing_style["bg"]
        self.preprocessing_frame.grid(column=col,
                                      row=0,
                                      padx=self.preprocessing_style["padx"],
                                      pady=self.preprocessing_style["pady"],
                                      sticky=stick_n2)

        [tkinter.Grid.rowconfigure(self.preprocessing_frame, i, weight=w)
         for i, w in enumerate(self.preprocessing_style["row_weights"])]
        [tkinter.Grid.columnconfigure(self.preprocessing_frame, i, weight=w)
         for i, w in enumerate(self.preprocessing_style["columns_weights"])]

        super().__init__(self.preprocessing_frame, module_name, font=font)
        self.module = "preprocessing"
        self.config = config

        self.show = tkinter.BooleanVar()
        # TODO fix state will reset and not load from the config
        # self.show.set(config[self.module]["state"])
        self.show.set(True)
        self.checkbox["variable"] = self.show
        self.checkbox["command"] = self.show_options

        self.obj_collection = []
        self.custom_key = {"save_directory": SimpleEntry(self.preprocessing_frame,
                                                         text="Save Directory: ",
                                                         row=1,
                                                         column=0,
                                                         _type=str,
                                                         _font=font),
                           "factor": ListEntry(self.preprocessing_frame,
                                               text="Rescaling Factor: ",
                                               row=2,
                                               column=0,
                                               font=font),
                           "order": SimpleEntry(self.preprocessing_frame,
                                                text="Interpolation: ",
                                                row=3,
                                                column=0,
                                                _type=int,
                                                _font=font),
                           "filter": FilterEntry(self.preprocessing_frame,
                                                 text="Filter: ",
                                                 row=4,
                                                 column=0,
                                                 font=font),
                           }

        self.show_options()


class UnetPredictionFrame(ModuleFramePrototype):
    def __init__(self, frame, config, col=0, module_name="preprocessing", font=None):
        self.prediction_frame = tkinter.Frame(frame)
        self.prediction_style = {"bg": "white",
                      "padx": 10,
                      "pady": 10,
                      "row_weights": [2, 1, 1, 1, 1],
                      "columns_weights": [1],
                      "height": 4,
                      }

        self.prediction_frame["bg"] = self.prediction_style["bg"]
        self.prediction_frame.grid(column=col,
                                    row=0,
                                    padx=self.prediction_style["padx"],
                                    pady=self.prediction_style["pady"],
                                    sticky=stick_n2)

        [tkinter.Grid.rowconfigure(self.prediction_frame, i, weight=w)
         for i, w in enumerate(self.prediction_style["row_weights"])]
        [tkinter.Grid.columnconfigure(self.prediction_frame, i, weight=w)
         for i, w in enumerate(self.prediction_style["columns_weights"])]

        super().__init__(self.prediction_frame, module_name, font=font)
        self.module = "unet_prediction"
        self.config = config

        self.show = tkinter.BooleanVar()
        # TODO fix state will reset and not load from the config
        # self.show.set(config[self.module]["state"])
        self.show.set(True)
        self.checkbox["variable"] = self.show
        self.checkbox["command"] = self.show_options

        self.obj_collection = []
        self.custom_key = {"model_name": MenuEntry(self.prediction_frame,
                                                   text="Model Name: ",
                                                   row=1,
                                                   column=0,
                                                   menu=list_models(),
                                                   default=config[self.module]["model_name"],
                                                   font=font),
                           "patch": ListEntry(self.prediction_frame,
                                              text="Patch Size: ",
                                              row=2,
                                              column=0,
                                              type=int,
                                              font=font),
                           "stride": ListEntry(self.prediction_frame,
                                               text="Stride: ",
                                               row=3,
                                               column=0,
                                               type=int,
                                               font=font),
                           "device": MenuEntry(self.prediction_frame,
                                               text="Device Type: ",
                                               row=4,
                                               column=0,
                                               menu=["cuda", "cpu"],
                                               default=config[self.module]["device"],
                                               font=font),
                           }

        self.show_options()


class SegmentationFrame(ModuleFramePrototype):
    def __init__(self, frame, config, col=0, module_name="segmentation", font=None):
        self.segmentation_frame = tkinter.Frame(frame)
        self.segmentation_style = {"bg": "white",
                      "padx": 10,
                      "pady": 10,
                      "row_weights": [2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      "columns_weights": [1],
                      "height": 4,
                      }

        self.segmentation_frame["bg"] = self.segmentation_style["bg"]
        self.segmentation_frame.grid(column=col,
                                      row=0,
                                      padx=self.segmentation_style["padx"],
                                      pady=self.segmentation_style["pady"],
                                      sticky=stick_n2)

        [tkinter.Grid.rowconfigure(self.segmentation_frame, i, weight=w)
         for i, w in enumerate(self.segmentation_style["row_weights"])]
        [tkinter.Grid.columnconfigure(self.segmentation_frame, i, weight=w)
         for i, w in enumerate(self.segmentation_style["columns_weights"])]

        super().__init__(self.segmentation_frame, module_name, font)
        self.module = "segmentation"
        self.config = config
        self.show = tkinter.BooleanVar()

        # TODO fix state will reset and not load from the config
        # self.show.set(config[self.module]["state"])
        self.show.set(True)
        self.checkbox["variable"] = self.show
        self.checkbox["command"] = self.show_options

        self.obj_collection = []
        self.custom_key = {"name": MenuEntry(self.segmentation_frame,
                                              text="Algorithm",
                                              row=1,
                                              column=0,
                                              menu={"MultiCut", "GASP", "MutexWS", "DtWatershed"},
                                              default=config[self.module]["name"],
                                              font=font),
                            "save_directory": SimpleEntry(self.segmentation_frame,
                                                         text="Save Directory: ",
                                                         row=2,
                                                         column=0,
                                                         _type=str,
                                                         _font=font),
                           "beta": SimpleEntry(self.segmentation_frame,
                                                        text="Beta: ",
                                                        row=3,
                                                        column=0,
                                                        _type=float,
                                                        _font=font),
                           "ws_2D": MenuEntry(self.segmentation_frame,
                                              text="Run WS in 2D: ",
                                              row=4,
                                              column=0,
                                              menu={"True", "False"},
                                              default=config[self.module]["ws_2D"],
                                              font=font),

                           "ws_threshold": SimpleEntry(self.segmentation_frame,
                                                   text="WS Threshold",
                                                   row=5,
                                                   column=0,
                                                   _type=float,
                                                   _font=font),

                           "ws_sigma": SimpleEntry(self.segmentation_frame,
                                                   text="WS Seeds Sigma: ",
                                                   row=6,
                                                   column=0,
                                                   _type=float,
                                                   _font=font),
                           "ws_w_sigma": SimpleEntry(self.segmentation_frame,
                                                     text="WS Boundary Sigma: ",
                                                     row=7,
                                                     column=0,
                                                     _type=float,
                                                     _font=font),
                           "ws_minsize": SimpleEntry(self.segmentation_frame,
                                                     text="WS Minimum Size: (voxels)",
                                                     row=8,
                                                     column=0,
                                                     _type=int,
                                                     _font=font),
                           "post_minsize": SimpleEntry(self.segmentation_frame,
                                                       text="Minimum Size: (voxels)",
                                                       row=9,
                                                       column=0,
                                                       _type=int,
                                                       _font=font),
                           }
        self.show_options()


class PostSegmentationFrame(ModuleFramePrototype):
    def __init__(self, frame, config, row=0, module_name="Segmentation Post Processing", font=None):
        self.post_frame = tkinter.Frame(frame)
        self.post_style = {"bg": "white",
                      "padx": 0,
                      "pady": 0,
                      "row_weights": [1, 1, 1, 1],
                      "columns_weights": [1],
                      "height": 4,
                      }

        self.post_frame["bg"] = self.post_style["bg"]
        self.font = font
        self.post_frame.grid(column=0,
                             row=row,
                             padx=self.post_style["padx"],
                             pady=self.post_style["pady"],
                             sticky=stick_all)

        [tkinter.Grid.rowconfigure(self.post_frame, i, weight=w)
         for i, w in enumerate(self.post_style["row_weights"])]
        [tkinter.Grid.columnconfigure(self.post_frame, i, weight=w)
         for i, w in enumerate(self.post_style["columns_weights"])]

        super().__init__(self.post_frame, module_name, font=font)
        self.module = "postprocessing"
        self.config = config["segmentation"]

        self.show = tkinter.BooleanVar()
        self.show.set(self.config[self.module]["state"])
        self.checkbox["variable"] = self.show
        self.checkbox["command"] = self.show_options

        self.obj_collection = []
        self.custom_key = {"tiff": MenuEntry(self.post_frame,
                                             text="Convert to tiff: ",
                                             row=1,
                                             column=0,
                                             menu=["True", "False"],
                                             font=font),
                           "factor": ListEntry(self.post_frame,
                                               text="Rescaling Factor: ",
                                               row=2,
                                               column=0,
                                               font=font),
                           }

        self.show_options()


class PostPredictionsFrame(ModuleFramePrototype):
    def __init__(self, frame, config, row=0, module_name="Prediction Post Processing", font=None):
        self.post_frame = tkinter.Frame(frame)
        self.post_style = {"bg": "white",
                      "padx": 0,
                      "pady": 0,
                      "row_weights": [1, 1, 1, 1],
                      "columns_weights": [1],
                      "height": 4,
                      }

        self.post_frame["bg"] = self.post_style["bg"]
        self.font = font

        self.post_frame.grid(column=0,
                             row=row,
                             padx=self.post_style["padx"],
                             pady=self.post_style["pady"],
                             sticky=stick_n2)

        [tkinter.Grid.rowconfigure(self.post_frame, i, weight=w)
         for i, w in enumerate(self.post_style["row_weights"])]
        [tkinter.Grid.columnconfigure(self.post_frame, i, weight=w)
         for i, w in enumerate(self.post_style["columns_weights"])]

        super().__init__(self.post_frame, module_name, font=font)
        self.module = "postprocessing"
        self.config = config["unet_prediction"]

        self.show = tkinter.BooleanVar()
        self.show.set(self.config[self.module]["state"])
        self.checkbox["variable"] = self.show
        self.checkbox["command"] = self.show_options

        self.obj_collection = []
        self.custom_key = {"tiff": MenuEntry(self.post_frame,
                                             text="Convert to tiff: ",
                                             row=1,
                                             column=0,
                                             menu=["True", "False"],
                                             font=font),
                           "factor": ListEntry(self.post_frame,
                                               text="Rescaling Factor: ",
                                               row=2,
                                               column=0,
                                               font=font),
                           "order": SimpleEntry(self.post_frame,
                                                text="Interpolation: ",
                                                row=3,
                                                column=0,
                                                _type=int,
                                                _font=font),
                           }

        self.show_options()


class PostFrame:
    def __init__(self, frame, config, col=0, font=None):
        self.post_frame = tkinter.Frame(frame)
        self.post_style = {"bg": "white",
                                   "padx": 10,
                                   "pady": 10,
                                   "row_weights": [1, 1],
                                   "columns_weights": [1],
                                   "height": 4,
                                   }

        self.post_frame["bg"] = self.post_style["bg"]
        self.font = font
        self.post_frame.grid(column=col,
                             row=0,
                             padx=self.post_style["padx"],
                             pady=self.post_style["pady"],
                             sticky=stick_n2)

        [tkinter.Grid.rowconfigure(self.post_frame, i, weight=w)
         for i, w in enumerate(self.post_style["row_weights"])]
        [tkinter.Grid.columnconfigure(self.post_frame, i, weight=w)
         for i, w in enumerate(self.post_style["columns_weights"])]

        ##
        config["segmentation"]["postprocessing"]["state"] = True
        config["unet_prediction"]["postprocessing"]["state"] = True

        self.post_pred_obj = PostPredictionsFrame(self.post_frame, config, row=0, font=font)
        # self.post_pred_obj.show.set(False)
        self.post_pred_obj.checkbox["bg"] = "white"

        self.post_seg_obj = PostSegmentationFrame(self.post_frame, config, row=1, font=font)
        # self.post_seg_obj.show.set(False)
        self.post_seg_obj.checkbox["bg"] = "white"

        self.post_frame.update()
        self.post_seg_obj.show_options()
        self.post_pred_obj.show_options()

        self.post_pred_obj.show.set(False)
        self.post_pred_obj.show_options()

        self.post_seg_obj.show.set(False)
        self.post_seg_obj.show_options()
