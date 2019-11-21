import tkinter
import yaml
from plantseg import raw2seg
import os
import sys
import webbrowser
from tkinter import font
from gui.gui_tools import Files2Process, report_error, StdoutRedirect, convert_rgb


class PlantSegApp:
    def __init__(self):
        # Init main app
        # App Setup ===================================================================================================
        # *--------------------------------------------------------------------------------------*
        # |                                    Browser Frame 0                                   |
        # *---------------------------*-----------------------------*----------------------------*
        # |     Config Frame 1.0      |      Config Frame 1.1       |      Config Frame 1.2      |
        # |                           |                             |                            |
        # |                           |                             |                            |
        # |                           |                             |                            |
        # *---------------------------*-----------------------------*----------------------------*
        # |                                      Run Frame 2                                     |
        # *--------------------------------------------------------------------------------------*
        # |                                      User Output                                     |
        # |                                                                                      |
        # |                                                                                      |
        # *--------------------------------------------------------------------------------------*

        # Load config all config
        self.app_config = self.load_app_config()
        self.plant_config_path, self.plantseg_config = self.load_config()

        # Init main app and configure
        self.plant_segapp = tkinter.Tk()
        self.plant_segapp.tk.call('tk', 'scaling', 1.0)

        self.plant_segapp.resizable(width=True, height=True)
        [tkinter.Grid.rowconfigure(self.plant_segapp, int(key), weight=value)
         for key, value in self.app_config["row_weights"].items()]
        [tkinter.Grid.columnconfigure(self.plant_segapp, int(key), weight=value)
         for key, value in self.app_config["columns_weights"].items()]
        self.plant_segapp.configure(bg=self.app_config["bg"])
        self.plant_segapp.title(self.app_config["title"])
        self.plant_segapp.update()

        # var
        self.stick_all = tkinter.N + tkinter.S + tkinter.E + tkinter.W

        self.pre_proc_obj, self.predictions_obj, self.segmentation_obj, self.post_obj = None, None, None, None
        self.file_to_process = None
        self.configuration_frame1 = None
        self.run_frame2, self.run_button = None, None
        self.out_text = None
        self.font_size = None
        self.font_bold, self.font = None, None

        # init blocks
        self.update_font(size=20)
        self.build_all()
        self.update_font(size=int(self.plant_segapp.winfo_width()/60))
        self.build_all()

        self.plant_segapp.protocol("WM_DELETE_WINDOW", self.close)
        sys.stdout = StdoutRedirect(self.out_text)
        self.plant_segapp.mainloop()

    def update_font(self, size=10, family="helvetica"):
        self.font_size = size
        self.font_bold = font.Font(family=family, size=self.font_size, weight="bold")
        self.font = font.Font(family=family, size=self.font_size)

    def build_all(self):
        self.build_menu()
        self.init_frame0()
        self.init_frame1()
        self.init_frame2()
        self.init_frame3()

    def build_menu(self):
        menubar = tkinter.Menu(self.plant_segapp)
        menubar["bg"] = convert_rgb(self.app_config["green"])
        filemenu = tkinter.Menu(menubar, tearoff=0)
        filemenu["bg"] = "white"
        filemenu.add_command(label="Open", command=self.open_config, font=self.font)
        filemenu.add_command(label="Save", command=self.save_config, font=self.font)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.close, font=self.font)
        menubar.add_cascade(label="File", menu=filemenu, font=self.font)

        preferencesmenu = tkinter.Menu(menubar, tearoff=0)
        preferencesmenu["bg"] = "white"

        preferencesmenu.add_command(label="Size +", command=self.size_up, font=self.font)
        preferencesmenu.add_command(label="Size -", command=self.size_down, font=self.font)
        menubar.add_cascade(label="Preferences", menu=preferencesmenu, font=self.font)

        helpmenu = tkinter.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Help Index", command=self.open_documentation, font=self.font)
        helpmenu.add_command(label="About...", font=self.font)
        helpmenu["bg"] = "white"
        menubar.add_cascade(label="Help", menu=helpmenu, font=self.font)

        self.plant_segapp.config(menu=menubar)

    def size_up(self):
        self.font_size += 2
        self.font_size = min(100, self.font_size)
        self.update_font(self.font_size)

        self.update_config()
        self.init_frame1()

    def size_down(self):
        self.font_size -= 2
        self.font_size = max(0, self.font_size)
        self.update_font(self.font_size)

        self.update_config()
        self.init_frame1()

    def init_frame0(self):
        # =============================================================================================================
        # Frame 0                                                                                                     #
        # =============================================================================================================
        browser_frame0 = tkinter.Frame(self.plant_segapp)
        browser_config = self.app_config["browser_frame"]
        [tkinter.Grid.rowconfigure(browser_frame0, int(key), weight=value)
         for key, value in browser_config["row_weights"].items()]
        [tkinter.Grid.columnconfigure(browser_frame0, int(key), weight=value)
         for key, value in browser_config["columns_weights"].items()]
        browser_frame0.grid(row=int(browser_config["row"]), column=int(browser_config["column"]), sticky=self.stick_all)
        browser_frame0["bg"] = browser_config["bg"]

        # Define file reader
        file_to_process = Files2Process(self.plantseg_config)

        x = tkinter.Label(browser_frame0, bg="white", text="File or Directory to Process", font=self.font_bold)

        x.grid(column=0, row=0, padx=10, pady=10, sticky=self.stick_all)

        x = tkinter.Entry(browser_frame0, textvar=file_to_process.files,
                          font=self.font)
        x.grid(column=1, row=0, padx=10, pady=10, sticky=self.stick_all)
        x = tkinter.Button(browser_frame0, bg="white", text="File",
                           command=file_to_process.browse_for_file, font=self.font_bold)
        x.grid(column=2, row=0, padx=0, pady=0, sticky=self.stick_all)
        x = tkinter.Button(browser_frame0, bg="white", text="Directory",
                           command=file_to_process.browse_for_directory, font=self.font_bold)
        x.grid(column=3, row=0, padx=0, pady=0, sticky=self.stick_all)
        self.file_to_process = file_to_process

    def init_frame1(self):
        # =============================================================================================================
        # Frame 1                                                                                                     #
        # =============================================================================================================
        configuration_frame1 = tkinter.Frame(self.plant_segapp)
        configuration_config = self.app_config["configuration_frame"]
        [tkinter.Grid.rowconfigure(configuration_frame1, int(key), weight=value)
         for key, value in configuration_config["row_weights"].items()]
        [tkinter.Grid.columnconfigure(configuration_frame1, int(key), weight=value)
         for key, value in configuration_config["columns_weights"].items()]
        configuration_frame1.grid(row=int(configuration_config["row"]),
                                  column=int(configuration_config["column"]),
                                  sticky=self.stick_all)
        configuration_frame1["highlightthickness"] = configuration_config["highlightthickness"]
        configuration_frame1["bg"] = configuration_config["bg"]

        self.configuration_frame1 = configuration_frame1

        (self.pre_proc_obj,
         self.predictions_obj,
         self.segmentation_obj,
         self.post_obj) = self.init_menus(self.plantseg_config)

    def init_frame2(self):
        # =============================================================================================================
        # Frame 2                                                                                                     #
        # =============================================================================================================
        run_frame2 = tkinter.Frame(self.plant_segapp)
        run_config = self.app_config["run_frame"]
        [tkinter.Grid.rowconfigure(run_frame2, int(key), weight=value)
         for key, value in run_config["row_weights"].items()]
        [tkinter.Grid.columnconfigure(run_frame2, int(key), weight=value)
         for key, value in run_config["columns_weights"].items()]
        run_frame2.grid(row=int(run_config["row"]),
                        column=int(run_config["column"]),
                        sticky=self.stick_all)
        run_frame2["highlightthickness"] = run_config["highlightthickness"]
        run_frame2["bg"] = run_config["bg"]

        x = tkinter.Button(run_frame2, bg=convert_rgb(self.app_config["green"]),
                           text="Docs", font=self.font_bold)
        x.grid(column=1, row=0, padx=10, pady=10, sticky=self.stick_all)
        x["command"] = self.open_documentation

        x = tkinter.Button(run_frame2, bg=convert_rgb(self.app_config["green"]),
                           text="Reset", font=self.font_bold)
        x.grid(column=2, row=0, padx=10, pady=10, sticky=self.stick_all)
        x["command"] = self.reset_config

        self.run_button = tkinter.Button(run_frame2, bg=convert_rgb(self.app_config["green"]),
                                         text="Run", command=self._run, font=self.font_bold)
        self.run_button.grid(column=3, row=0, padx=10, pady=10, sticky=self.stick_all)

        self.run_frame2 = run_frame2

    def init_frame3(self):
        # ============================================================================================================
        # Frame 3                                                                                                     #
        # ============================================================================================================
        out_frame3 = tkinter.Frame(self.plant_segapp)
        out_config = self.app_config["out_frame"]
        [tkinter.Grid.rowconfigure(out_frame3, int(key), weight=value)
         for key, value in out_config["row_weights"].items()]
        [tkinter.Grid.columnconfigure(out_frame3, int(key), weight=value)
         for key, value in out_config["columns_weights"].items()]

        out_frame3.grid(row=int(out_config["row"]),
                        column=int(out_config["column"]),
                        sticky=self.stick_all)
        out_frame3["highlightthickness"] = out_config["highlightthickness"]
        out_frame3["bg"] = out_config["bg"]

        out_text = tkinter.Text(out_frame3, height=6)
        scroll_bar = tkinter.Scrollbar(out_frame3)
        scroll_bar["bg"] = out_config["bg"]

        scroll_bar.config(command=out_text.yview)
        out_text.config(yscrollcommand=scroll_bar.set)
        out_text.grid(column=0, row=0, padx=10, pady=10, sticky=self.stick_all)
        out_text.configure(state='disabled')
        out_text.configure(font=self.font)
        scroll_bar.grid(column=1, row=0, padx=10, pady=10, sticky=self.stick_all)
        self.out_text = out_text

    @staticmethod
    def load_config(name="config_custom.yaml"):
        """Load the last (or if not possible a standard) config"""
        plant_config_path = os.path.join(os.path.dirname(os.path.abspath('__file__')), name)
        if os.path.exists(plant_config_path):
            plantseg_config = yaml.load(open(plant_config_path, 'r'), Loader=yaml.FullLoader)
        else:
            plant_config_path = os.path.join(os.path.dirname(os.path.abspath('__file__')), "config.yaml")
            plantseg_config = yaml.load(open(plant_config_path, 'r'), Loader=yaml.FullLoader)

        return plant_config_path, plantseg_config

    @staticmethod
    def load_app_config(config="gui_configuration.yaml"):
        """Load gui style config"""
        conf_path = os.path.join(os.path.dirname(os.path.abspath('__file__')), "gui", config)
        app_config = yaml.load(open(conf_path, 'r'), Loader=yaml.FullLoader)["plant_segapp"]
        return app_config

    def init_menus(self, config):
        """ Initialize menu entries"""
        from gui.gui_widgets import PreprocessingFrame, UnetPredictionFrame, SegmentationFrame, PostFrame
        pre_proc_obj = PreprocessingFrame(self.configuration_frame1, config,
                                          col=0, module_name="Data Pre-Processing",
                                          font=self.font)
        predictions_obj = UnetPredictionFrame(self.configuration_frame1, config,
                                              col=1, module_name="3D - Unet",
                                              font=self.font)
        segmentation_obj = SegmentationFrame(self.configuration_frame1, config,
                                             col=2, module_name="Segmentation",
                                             font=self.font)
        post_obj = PostFrame(self.configuration_frame1, config,
                             col=3, font=self.font)
        return pre_proc_obj, predictions_obj, segmentation_obj, post_obj

    @staticmethod
    def open_documentation():
        """Open git page on the default browser"""
        webbrowser.open("https://github.com/hci-unihd/plant-seg")

    def close(self):
        """Thi function let the user decide if saving  the current config"""
        def close_action():
            """simply close the app"""
            self.plant_segapp.destroy()
            popup.destroy()

        def close_action_and_save():
            """save current configuration and close the app"""
            self.update_config()
            plant_config_path = os.path.join(os.path.dirname(os.path.abspath('__file__')), "config_custom.yaml")
            with open(plant_config_path, "w") as f:
                yaml.dump(self.plantseg_config, f)

            close_action()

        # Main Popup window
        popup = tkinter.Tk()
        popup.title("Quit")
        popup.configure(bg=self.app_config["bg"])
        tkinter.Grid.rowconfigure(popup, 0, weight=1)
        tkinter.Grid.columnconfigure(popup, 0, weight=1)
        tkinter.Grid.rowconfigure(popup, 1, weight=1)

        x = tkinter.Label(popup, bg="white", text="Save Current Config?")
        x.grid(column=0, row=0, padx=10, pady=10, sticky=self.stick_all)

        # Actions
        button_frame = tkinter.Frame(popup)
        button_frame.configure(bg=self.app_config["bg"])
        button_frame.grid(column=0, row=1)
        x = tkinter.Button(button_frame, bg="white", text="No", command=close_action)
        x.grid(column=0, row=0, padx=10, pady=10, sticky=self.stick_all)

        x = tkinter.Button(button_frame, bg="white", text="Yes", command=close_action_and_save)
        x.grid(column=1, row=0, padx=10, pady=10, sticky=self.stick_all)

    def reset_config(self):
        """ reset to default config"""
        plant_config_path = os.path.join(os.path.dirname(os.path.abspath('__file__')), "config.yaml")
        plantseg_config = yaml.load(open(plant_config_path, 'r'), Loader=yaml.FullLoader)

        (self.pre_proc_obj,
         self.predictions_obj,
         self.segmentation_obj,
         self.post_obj) = self.init_menus(plantseg_config)
        self.plantseg_config = plantseg_config

    def open_config(self):
        """ open new config"""
        plant_config_path = tkinter.filedialog.askopenfilename(initialdir=os.path.expanduser("~"),
                                                               title="Select file",
                                                               filetypes=(("yaml files", "*.yaml"),
                                                                          ("yaml files", "*.yml")))
        plantseg_config = yaml.load(open(plant_config_path, 'r'), Loader=yaml.FullLoader)

        (self.pre_proc_obj,
         self.predictions_obj,
         self.segmentation_obj,
         self.post_obj) = self.init_menus(plantseg_config)
        self.plantseg_config = plantseg_config

    def update_config(self):
        """ create from gui an updated yaml dictionary"""
        plantseg_config = yaml.load(open(self.plant_config_path, 'r'), Loader=yaml.FullLoader)

        plantseg_config["path"] = self.file_to_process.files.get()
        # Update config
        plantseg_config = self.pre_proc_obj.check_and_update_config(plantseg_config,
                                                                    dict1="preprocessing",
                                                                    dict2=False)

        plantseg_config = self.post_obj.post_pred_obj.check_and_update_config(plantseg_config,
                                                                              dict1="unet_prediction",
                                                                              dict2="postprocessing")
        plantseg_config = self.predictions_obj.check_and_update_config(plantseg_config,
                                                                       dict1="unet_prediction",
                                                                       dict2=False)

        plantseg_config = self.post_obj.post_pred_obj.check_and_update_config(plantseg_config,
                                                                              dict1="segmentation",
                                                                              dict2="postprocessing")
        plantseg_config = self.segmentation_obj.check_and_update_config(plantseg_config,
                                                                        dict1="segmentation",
                                                                        dict2=False)
        self.plantseg_config = plantseg_config

    def save_config(self):
        """ save yaml from current entries in the gui"""
        self.update_config()
        save_path = tkinter.filedialog.asksaveasfilename(initialdir=os.path.expanduser("~"),
                                                         defaultextension="yaml",
                                                         filetypes=(("yaml files", "*.yaml"),))
        with open(save_path, "w") as f:
            yaml.dump(self.plantseg_config, f)

    def _run(self):
        """ create a yaml config from the gui and run the pipeline accordingly"""
        self.run_button["state"] = "disabled"
        self.update_config()
        self.plant_segapp.update()

        print(f"Final config:")
        for key in self.plantseg_config.keys():
            print(f"{key}: {self.plantseg_config[key]}")

        #try:
        raw2seg(self.plantseg_config)
        #except Exception as e:
        #    report_error(e)

        self.run_button["state"] = "normal"


if __name__ == "__main__":
    PlantSegApp()
