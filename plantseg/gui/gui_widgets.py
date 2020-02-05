import tkinter

from plantseg.gui import convert_rgb, list_models
from plantseg.gui import stick_all, stick_new
from plantseg.gui.gui_tools import ListEntry, SimpleEntry, FilterEntry, RescaleEntry, MenuEntry


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
            self.update_config(config, module)

            for obj in self.obj_collection:
                obj.grid_forget()

        return config

    def check_and_update_config(self, config, dict_key):
        if self.show.get():
            config[dict_key]["state"] = True
            config = self.update_config(config, dict_key)
        else:
            config[dict_key]["state"] = False

        return config

    def update_config(self, config, dict_key):
        print(dict_key)
        print(config[dict_key])
        for key, obj in self.custom_key.items():
            if key in config[dict_key]:
                if isinstance(obj, MenuEntry):
                    str_value = obj.tk_value.get()
                    str_value = True if str_value == "True" else str_value
                    str_value = False if str_value == "False" else str_value
                    config[dict_key][key] = str_value

                elif isinstance(obj, SimpleEntry):
                    str_value = obj.tk_value.get()
                    config[dict_key][key] = obj.type(str_value)

                elif isinstance(obj, FilterEntry):
                    if obj.tk_value[0].get():
                        config[dict_key]["filter"]["state"] = True
                        config[dict_key]["filter"]["type"] = obj.tk_value[1].get()
                        config[dict_key]["filter"]["param"] = float(obj.tk_value[2].get())
                    else:
                        config[dict_key]["filter"]["state"] = False

                elif isinstance(obj, RescaleEntry):
                    values = [obj.tk_value[0].get(), obj.tk_value[1].get(), obj.tk_value[2].get()]
                    values = [obj.type(values[0]), obj.type(values[1]), obj.type(values[2])]
                    ivalues = [1 / values[0], 1 / values[1], 1 / values[2]]

                    config[dict_key]["factor"] = values
                    config["cnn_postprocessing"]["factor"] = ivalues
                    config["segmentation_postprocessing"]["factor"] = ivalues

                elif isinstance(obj, ListEntry):
                    values = [obj.tk_value[0].get(), obj.tk_value[1].get(), obj.tk_value[2].get()]
                    values = [obj.type(values[0]), obj.type(values[1]), obj.type(values[2])]
                    config[dict_key][key] = values
        print("after", config[dict_key])
        return config

    def show_options(self):
        self.config = self._show_options(self.config, self.module)


class PreprocessingFrame(ModuleFramePrototype):
    def __init__(self, frame, config, col=0, module_name="preprocessing", font=None, show_all=True):
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
                                      sticky=stick_new)

        [tkinter.Grid.rowconfigure(self.preprocessing_frame, i, weight=w)
         for i, w in enumerate(self.preprocessing_style["row_weights"])]
        [tkinter.Grid.columnconfigure(self.preprocessing_frame, i, weight=w)
         for i, w in enumerate(self.preprocessing_style["columns_weights"])]

        super().__init__(self.preprocessing_frame, module_name, font=font)
        self.module = "preprocessing"
        self.config = config

        self.show = tkinter.BooleanVar()
        if show_all:
            self.show.set(True)
        else:
            self.show.set(self.config[self.module]["state"])

        self.checkbox["variable"] = self.show
        self.checkbox["command"] = self.show_options

        self.obj_collection = []
        self.custom_key = {"save_directory": SimpleEntry(self.preprocessing_frame,
                                                         text="Save Directory: ",
                                                         row=1,
                                                         column=0,
                                                         _type=str,
                                                         _font=font),
                           "factor": RescaleEntry(self.preprocessing_frame,
                                                  text="Rescaling: ",
                                                  row=2,
                                                  column=0,
                                                  font=font),
                           "order": MenuEntry(self.preprocessing_frame,
                                              text="Interpolation: ",
                                              row=3,
                                              column=0,
                                              menu=[0, 1, 2],
                                              default=2),
                           "filter": FilterEntry(self.preprocessing_frame,
                                                 text="Filter: ",
                                                 row=4,
                                                 column=0,
                                                 font=font),
                           }

        self.show_options()


class UnetPredictionFrame(ModuleFramePrototype):
    def __init__(self, frame, config, col=0, module_name="preprocessing", font=None, show_all=True):
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
                                   sticky=stick_new)

        [tkinter.Grid.rowconfigure(self.prediction_frame, i, weight=w)
         for i, w in enumerate(self.prediction_style["row_weights"])]
        [tkinter.Grid.columnconfigure(self.prediction_frame, i, weight=w)
         for i, w in enumerate(self.prediction_style["columns_weights"])]

        super().__init__(self.prediction_frame, module_name, font=font)
        self.module = "cnn_prediction"
        self.config = config
        self.show = tkinter.BooleanVar()

        if show_all:
            self.show.set(True)
        else:
            self.show.set(self.config[self.module]["state"])

        self.checkbox["variable"] = self.show
        self.checkbox["command"] = self.show_options

        self.obj_collection = []
        self.custom_key = {"model_name": MenuEntry(self.prediction_frame,
                                                   text="Model Name: ",
                                                   row=1,
                                                   column=0,
                                                   menu=list_models(),
                                                   default=config[self.module]["model_name"],
                                                   is_model=True,
                                                   font=font),
                           "patch": ListEntry(self.prediction_frame,
                                              text="Patch Size: ",
                                              row=2,
                                              column=0,
                                              type=int,
                                              font=font),
                           "stride": MenuEntry(self.prediction_frame,
                                               text="Stride: ",
                                               row=3,
                                               column=0,
                                               menu=["Accurate (slowest)", "Balanced", "Draft (fastest)"],
                                               default=config[self.module]["stride"],
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
    def __init__(self, frame, config, col=0, module_name="segmentation", font=None, show_all=True):
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
                                     sticky=stick_new)

        [tkinter.Grid.rowconfigure(self.segmentation_frame, i, weight=w)
         for i, w in enumerate(self.segmentation_style["row_weights"])]
        [tkinter.Grid.columnconfigure(self.segmentation_frame, i, weight=w)
         for i, w in enumerate(self.segmentation_style["columns_weights"])]

        super().__init__(self.segmentation_frame, module_name, font)
        self.module = "segmentation"
        self.config = config
        self.show = tkinter.BooleanVar()

        if show_all:
            self.show.set(True)
        else:
            self.show.set(self.config[self.module]["state"])

        self.checkbox["variable"] = self.show
        self.checkbox["command"] = self.show_options

        self.obj_collection = []
        self.custom_key = {"name": MenuEntry(self.segmentation_frame,
                                              text="Algorithm: ",
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
                                                   text="WS Threshold ",
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
                                                     text="WS Minimum Size: (voxels) ",
                                                     row=8,
                                                     column=0,
                                                     _type=int,
                                                     _font=font),
                           "post_minsize": SimpleEntry(self.segmentation_frame,
                                                     text="Minimum Size: (voxels) ",
                                                       row=9,
                                                       column=0,
                                                       _type=int,
                                                       _font=font),
                           }

        self.show_options()


class PostSegmentationFrame(ModuleFramePrototype):
    def __init__(self, frame, config, row=0, module_name="Segmentation Post Processing", font=None, show_all=True):
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
        self.module = "cnn_postprocessing"
        self.config = config

        self.show = tkinter.BooleanVar()
        if show_all:
            self.show.set(True)
        else:
            self.show.set(self.config[self.module]["state"])

        self.checkbox["variable"] = self.show
        self.checkbox["command"] = self.show_options

        self.obj_collection = []
        self.custom_key = {"tiff": MenuEntry(self.post_frame,
                                             text="Convert to tiff: ",
                                             row=1,
                                             column=0,
                                             menu=["True", "False"],
                                             default=self.config[self.module]["tiff"],
                                             font=font),
                           }

        self.show_options()


class PostPredictionsFrame(ModuleFramePrototype):
    def __init__(self, frame, config, row=0, module_name="Prediction Post Processing", font=None, show_all=True):
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
                             sticky=stick_new)

        [tkinter.Grid.rowconfigure(self.post_frame, i, weight=w)
         for i, w in enumerate(self.post_style["row_weights"])]
        [tkinter.Grid.columnconfigure(self.post_frame, i, weight=w)
         for i, w in enumerate(self.post_style["columns_weights"])]

        super().__init__(self.post_frame, module_name, font=font)
        self.module = "cnn_postprocessing"
        self.config = config

        self.show = tkinter.BooleanVar()
        if show_all:
            self.show.set(True)
        else:
            self.show.set(self.config[self.module]["state"])

        self.checkbox["variable"] = self.show
        self.checkbox["command"] = self.show_options

        self.obj_collection = []
        self.custom_key = {"tiff": MenuEntry(self.post_frame,
                                             text="Convert to tiff: ",
                                             row=1,
                                             column=0,
                                             menu=["True", "False"],
                                             default=self.config[self.module]["tiff"],
                                             font=font),
                           "output_type": MenuEntry(self.post_frame,
                                                    text="Cast Predictions: ",
                                                    row=4,
                                                    column=0,
                                                    menu=["data_uint8", "data_float32"],
                                                    default=config[self.module]["output_type"],
                                                    font=font)
                           }

        self.show_options()


class PostFrame:
    def __init__(self, frame, config, col=0, font=None, show_all=True):
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
                             sticky=stick_new)

        [tkinter.Grid.rowconfigure(self.post_frame, i, weight=w)
         for i, w in enumerate(self.post_style["row_weights"])]
        [tkinter.Grid.columnconfigure(self.post_frame, i, weight=w)
         for i, w in enumerate(self.post_style["columns_weights"])]

        # init frames
        self.post_pred_obj = PostPredictionsFrame(self.post_frame, config, row=0, font=font, show_all=True)
        self.post_seg_obj = PostSegmentationFrame(self.post_frame, config, row=1, font=font, show_all=True)
