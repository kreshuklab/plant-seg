import logging
import os
import queue
import sys
import tkinter
import webbrowser
from tkinter import font

import yaml

from plantseg import plantseg_global_path, PLANTSEG_MODELS_DIR
from plantseg.gui import convert_rgb
from plantseg.gui.gui_tools import Files2Process, report_error, version_popup
from plantseg.pipeline import gui_logger
from plantseg.pipeline.executor import PipelineExecutor
from plantseg.pipeline.utils import QueueHandler


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

        # Load app config
        self.app_config = self.load_app_config()
        self.plant_config_path, self.plantseg_config = self.load_config()

        # Init main app and configure
        self.plant_segapp = tkinter.Tk()
        self.plant_segapp.tk.call('tk', 'scaling', 1.0)

        # Set icon
        icon_path = self.get_icon_path()
        icon = tkinter.PhotoImage(file=icon_path)
        self.plant_segapp.tk.call('wm', 'iconphoto', self.plant_segapp._w, icon)

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

        # create pipeline executor; hardcode max_workers and max_size for now
        self.pipeline_executor = PipelineExecutor(max_workers=1, max_size=1)
        # init blocks
        self.update_font(size=self.app_config["fontsize"])
        self.build_all()

        self.plant_segapp.protocol("WM_DELETE_WINDOW", self.close)
        self.plant_segapp.mainloop()

    def update_font(self, size=10, family="helvetica"):
        self.font_size = size
        self.font_bold = font.Font(family=family, size=self.font_size, weight="bold")
        self.font = font.Font(family=family, size=self.font_size)
        self.app_config["fontsize"] = self.font_size

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
        filemenu.add_command(label="Restart", command=self.restart_program, font=self.font)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.close, font=self.font)
        menubar.add_cascade(label="File", menu=filemenu, font=self.font)

        preferencesmenu = tkinter.Menu(menubar, tearoff=0)
        preferencesmenu["bg"] = "white"

        preferencesmenu.add_command(label="Font Size +", command=self.size_up, font=self.font)
        preferencesmenu.add_command(label="Font Size -", command=self.size_down, font=self.font)
        menubar.add_cascade(label="Preferences", menu=preferencesmenu, font=self.font)

        helpmenu = tkinter.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Help Index",
                             command=self.open_documentation_index, font=self.font)
        helpmenu.add_command(label="PlantSeg Overview",
                             command=self.open_documentation_general, font=self.font)
        helpmenu.add_command(label="Data Pre-Processing",
                             command=self.open_documentation_preprocessing, font=self.font)
        helpmenu.add_command(label="3D-Unet",
                             command=self.open_documentation_3dunet, font=self.font)
        helpmenu.add_command(label="Segmentation",
                             command=self.open_documentation_segmentation, font=self.font)
        helpmenu.add_command(label="Data Post-Processing",
                             command=self.open_documentation_preprocessing, font=self.font)
        helpmenu.add_separator()
        helpmenu.add_command(label="About...", command=version_popup, font=self.font)

        helpmenu["bg"] = "white"
        menubar.add_cascade(label="Help", menu=helpmenu, font=self.font)

        self.plant_segapp.config(menu=menubar)

    def init_frame0(self):
        # =============================================================================================================
        # Frame 0                                                                                                     #
        # =============================================================================================================
        browser_frame0 = tkinter.Frame(self.plant_segapp)
        browser_config = self.app_config["browser_frame"]

        self.config_row_column(browser_frame0, browser_config)

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
        self.config_row_column(configuration_frame1, configuration_config)

        configuration_frame1.grid(row=int(configuration_config["row"]),
                                  column=int(configuration_config["column"]),
                                  sticky=self.stick_all)
        configuration_frame1["highlightthickness"] = configuration_config["highlightthickness"]
        configuration_frame1["bg"] = configuration_config["bg"]

        self.configuration_frame1 = configuration_frame1

        (self.pre_proc_obj,
         self.predictions_obj,
         self.segmentation_obj,
         self.post_obj) = self.init_menus(show_all=True)

    def init_frame2(self):
        # =============================================================================================================
        # Frame 2                                                                                                     #
        # =============================================================================================================
        run_frame2 = tkinter.Frame(self.plant_segapp)
        run_config = self.app_config["run_frame"]

        self.config_row_column(run_frame2, run_config)

        run_frame2.grid(row=int(run_config["row"]),
                        column=int(run_config["column"]),
                        sticky=self.stick_all)
        run_frame2["highlightthickness"] = run_config["highlightthickness"]
        run_frame2["bg"] = run_config["bg"]

        x = tkinter.Button(run_frame2, bg=convert_rgb(self.app_config["green"]),
                           text="PlantSeg Introduction", font=self.font_bold)
        x.grid(column=3, row=0, padx=10, pady=10, sticky=self.stick_all)
        x["command"] = self.open_documentation_index

        x = tkinter.Button(run_frame2, bg=convert_rgb(self.app_config["green"]),
                           text="Reset Parameters", font=self.font_bold)
        x.grid(column=4, row=0, padx=10, pady=10, sticky=self.stick_all)
        x["command"] = self.reset_config

        self.run_button = tkinter.Button(run_frame2, bg=convert_rgb(self.app_config["green"]),
                                         text="Run", command=self._run, font=self.font_bold)
        self.run_button.grid(column=5, row=0, padx=10, pady=10, sticky=self.stick_all)

        self.run_frame2 = run_frame2

    def init_frame3(self):
        # ============================================================================================================
        # Frame 3                                                                                                     #
        # ============================================================================================================
        out_frame3 = tkinter.Frame(self.plant_segapp)
        out_config = self.app_config["out_frame"]
        self.config_row_column(out_frame3, out_config)

        out_frame3.grid(row=int(out_config["row"]),
                        column=int(out_config["column"]),
                        sticky=self.stick_all)
        out_frame3["highlightthickness"] = out_config["highlightthickness"]
        out_frame3["bg"] = out_config["bg"]

        out_text = tkinter.Text(out_frame3, height=12)
        scroll_bar = tkinter.Scrollbar(out_frame3)
        scroll_bar["bg"] = out_config["bg"]

        scroll_bar.config(command=out_text.yview)
        out_text.config(yscrollcommand=scroll_bar.set)
        out_text.grid(column=0, row=0, padx=10, pady=10, sticky=self.stick_all)
        out_text.configure(font=self.font)
        scroll_bar.grid(column=1, row=0, padx=10, pady=10, sticky=self.stick_all)

        # configure log level display
        out_text.tag_config('INFO', foreground='black')
        out_text.tag_config('DEBUG', foreground='gray')
        out_text.tag_config('WARNING', foreground='orange')
        out_text.tag_config('ERROR', foreground='red')
        out_text.tag_config('CRITICAL', foreground='red', underline=1)

        out_text.configure(state='disabled')

        # TODO: refactor
        # Create a logging handler using a queue
        self.out_frame3 = out_frame3
        self.log_queue = queue.Queue()
        self.queue_handler = QueueHandler(self.log_queue)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        self.queue_handler.setFormatter(formatter)
        gui_logger.addHandler(self.queue_handler)
        # Start polling messages from the queue
        self.out_frame3.after(100, self.poll_log_queue)

        self.out_text = out_text

    # Copied from https://github.com/beenje/tkinter-logging-text-widget/blob/master/main.py
    def poll_log_queue(self):
        # Check every 100ms if there is a new message in the queue to display
        while True:
            try:
                record = self.log_queue.get(block=False)
            except queue.Empty:
                break
            else:
                self.display(record)
        self.out_frame3.after(100, self.poll_log_queue)

    def display(self, record):
        msg = self.queue_handler.format(record)
        if record.levelname in ['ERROR', 'CRITICAL']:
            # show pop-up in case of error
            report_error(msg)
        else:
            self.out_text.configure(state='normal')
            self.out_text.insert(tkinter.END, msg + '\n', record.levelname)
            self.out_text.configure(state='disabled')
            # Autoscroll to the bottom
            self.out_text.yview(tkinter.END)

    # End init modules ========= Begin Config Read/Write
    @staticmethod
    def get_model_path():
        # Working directory path + relative dir structure to yaml file
        config_path = os.path.join(plantseg_global_path, "resources", "models_zoo.yaml")
        return config_path

    @staticmethod
    def get_last_config_path(name="config_gui_last.yaml"):
        # Working directory path + relative dir structure to yaml file
        config_path = os.path.join(os.path.expanduser("~"), PLANTSEG_MODELS_DIR, "configs", name)
        return config_path

    @staticmethod
    def get_app_config_path(name="gui_configuration.yaml"):
        # Working directory path + relative dir structure to yaml file
        config_path = os.path.join(plantseg_global_path, "resources", name)
        return config_path

    @staticmethod
    def get_icon_path(name="FOR2581_Logo_FINAL_no_text.png"):
        # Working directory path + relative dir structure to yaml file
        icon_path = os.path.join(plantseg_global_path, "resources", name)
        return icon_path

    def load_config(self, name="config_gui_last.yaml"):
        """Load the last (or if not possible a standard) config"""
        plant_config_path = self.get_last_config_path(name)

        if os.path.exists(plant_config_path):
            plantseg_config = yaml.load(open(plant_config_path, 'r'), Loader=yaml.FullLoader)
        else:
            # Do not modify this location
            plant_config_path = os.path.join(plantseg_global_path,
                                             "resources",
                                             "config_gui_template.yaml")
            plantseg_config = yaml.load(open(plant_config_path, 'r'), Loader=yaml.FullLoader)

        return plant_config_path, plantseg_config

    def load_app_config(self, config="gui_configuration.yaml"):
        """Load gui style config"""
        conf_path = self.get_app_config_path(config)
        app_config = yaml.load(open(conf_path, 'r'), Loader=yaml.FullLoader)["plant_segapp"]
        return app_config

    def reset_config(self):
        """ reset to default config, do not change path"""
        plant_config_path = os.path.join(plantseg_global_path,
                                         "resources",
                                         "config_gui_template.yaml")
        self.plantseg_config = yaml.load(open(plant_config_path, 'r'), Loader=yaml.FullLoader)

        (self.pre_proc_obj,
         self.predictions_obj,
         self.segmentation_obj,
         self.post_obj) = self.init_menus()

    def open_config(self):
        """ open new config"""
        default_start = os.path.join(os.path.expanduser("~"), PLANTSEG_MODELS_DIR, "configs")
        os.makedirs(default_start, exist_ok=True)
        plant_config_path = tkinter.filedialog.askopenfilename(initialdir=default_start,
                                                               title="Select file",
                                                               filetypes=(("yaml files", "*.yaml"),
                                                                          ("yaml files", "*.yml")))
        if len(plant_config_path) > 0:
            self.plantseg_config = yaml.load(open(plant_config_path, 'r'), Loader=yaml.FullLoader)
            (self.pre_proc_obj,
             self.predictions_obj,
             self.segmentation_obj,
             self.post_obj) = self.init_menus()

    def save_config(self):
        """ save yaml from current entries in the gui"""
        self.update_config()
        default_start = os.path.join(os.path.expanduser("~"), PLANTSEG_MODELS_DIR, "configs")
        os.makedirs(default_start, exist_ok=True)

        save_path = tkinter.filedialog.asksaveasfilename(initialdir=default_start,
                                                         defaultextension=".yaml",
                                                         filetypes=(("yaml files", "*.yaml"),))
        if len(save_path) > 0:
            with open(save_path, "w") as f:
                yaml.dump(self.plantseg_config, f)

    # End config Read/Write ========= Begin Others
    def init_menus(self, show_all=True):
        """ Initialize menu entries from config"""
        from ..gui.gui_widgets import PreprocessingFrame, UnetPredictionFrame, SegmentationFrame, PostFrame
        pre_proc_obj = PreprocessingFrame(self.configuration_frame1, self.plantseg_config,
                                          col=0, module_name="Data Pre-Processing",
                                          font=self.font, show_all=show_all)
        predictions_obj = UnetPredictionFrame(self.configuration_frame1, self.plantseg_config,
                                              col=1, module_name="3D-Unet",
                                              font=self.font, show_all=show_all)
        segmentation_obj = SegmentationFrame(self.configuration_frame1, self.plantseg_config,
                                             col=2, module_name="Segmentation",
                                             font=self.font, show_all=show_all)
        post_obj = PostFrame(self.configuration_frame1, self.plantseg_config,
                             col=3, font=self.font, show_all=show_all)

        return pre_proc_obj, predictions_obj, segmentation_obj, post_obj

    @staticmethod
    def config_row_column(frame, config):
        _ = [tkinter.Grid.rowconfigure(frame, int(key), weight=value)
             for key, value in config["row_weights"].items()]
        _ = [tkinter.Grid.columnconfigure(frame, int(key), weight=value)
             for key, value in config["columns_weights"].items()]

    @staticmethod
    def open_documentation_index():
        """Open git page on the default browser"""
        webbrowser.open("https://github.com/hci-unihd/plant-seg/tree/master/Documentation-GUI")

    @staticmethod
    def open_documentation_general():
        """Open git page on the default browser"""
        webbrowser.open("https://github.com/hci-unihd/plant-seg/blob/master/Documentation-GUI/General_gui.md")

    @staticmethod
    def open_documentation_preprocessing():
        """Open git page on the default browser"""
        webbrowser.open("https://github.com/hci-unihd/plant-seg/blob/master/Documentation-GUI/Data-Processing.md")

    @staticmethod
    def open_documentation_3dunet():
        """Open git page on the default browser"""
        webbrowser.open("https://github.com/hci-unihd/plant-seg/blob/master/Documentation-GUI/Predictions.md")

    @staticmethod
    def open_documentation_segmentation():
        """Open git page on the default browser"""
        webbrowser.open("https://github.com/hci-unihd/plant-seg/blob/master/Documentation-GUI/Segmentation.md")

    @staticmethod
    def open_postprocessing():
        """Open git page on the default browser"""
        webbrowser.open("https://github.com/hci-unihd/plant-seg/blob/master/Documentation-GUI/Data-Processing.md")

    def size_up(self):
        """ adjust font size in the main widget"""
        self.font_size += 2
        self.font_size = min(100, self.font_size)
        self.update_font(self.font_size)

        self.update_config()
        self.build_all()

    def size_down(self):
        """ adjust font size in the main widget"""
        self.font_size -= 2
        self.font_size = max(0, self.font_size)
        self.update_font(self.font_size)

        self.update_config()
        self.build_all()

    @staticmethod
    def restart_program():
        """Restarts the current program.
        Note: this function does not return. Any cleanup action (like
        saving data) must be done before calling this function.
        source: https://www.daniweb.com/programming/software-development/code/260268/restart-your-python-program
        """
        python = sys.executable
        os.execl(python, python, *sys.argv)

    def close(self):
        """Thi function let the user decide if saving  the current config"""

        def close_action():
            """simply close the app"""
            # shutdown pipeline executor without waiting for the current task to finish
            self.pipeline_executor.shutdown(wait=False)

            self.plant_segapp.destroy()
            popup.destroy()

        def close_action_and_save():
            """save current configuration and close the app"""
            self.update_config()

            plant_config_path = self.get_last_config_path()

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

    def update_config(self):
        """ create from gui an updated yaml dictionary"""

        # open a template config
        plantseg_config = yaml.load(open(self.plant_config_path, 'r'), Loader=yaml.FullLoader)

        # fill with modules input
        plantseg_config["path"] = self.file_to_process.files.get()

        plantseg_config = self.pre_proc_obj.check_and_update_config(plantseg_config,
                                                                    dict_key="preprocessing")

        plantseg_config = self.predictions_obj.check_and_update_config(plantseg_config,
                                                                       dict_key="cnn_prediction")

        plantseg_config = self.post_obj.post_pred_obj.check_and_update_config(plantseg_config,
                                                                              dict_key="cnn_postprocessing")

        plantseg_config = self.segmentation_obj.check_and_update_config(plantseg_config,
                                                                        dict_key="segmentation")

        plantseg_config = self.post_obj.post_seg_obj.check_and_update_config(plantseg_config,
                                                                             dict_key="segmentation_postprocessing")
        # Save plantseg_config
        self.plantseg_config = plantseg_config

    def _run(self):
        """ create a yaml config from the gui and run the pipeline accordingly"""
        # Disable run button to avoid multiple actions.
        self.run_button["state"] = "disabled"

        # Update config file from gui's menu
        self.update_config()
        self.plant_segapp.update()

        # Run the pipeline
        try:
            if not self.pipeline_executor.full():
                # execute the pipeline
                self.pipeline_executor.submit(self.plantseg_config)
            else:
                report_error("Cannot execute another task. Wait for the current segmentation pipeline to finish.")
        except Exception as e:
            # If an error occur generate a popup
            report_error(e)

        # Enable the run button
        self.run_button["state"] = "normal"
