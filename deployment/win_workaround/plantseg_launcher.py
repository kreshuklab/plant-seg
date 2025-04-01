import os
import subprocess
import sys
import zipfile

import requests
import yaml

__version__ = "0.0.1"

# Create configs directory at startup
PLANTSEG_MODELS_DIR = ".plantseg_models"
PLANTSEG_BIN_ZIP = "plant-seg.zip"
PLANTSEG_BIN_DIR = "plant-seg-win"

plantseg_win_urls = {
    "gpu": {
        "description": "Only suggested if a NVIDIA Gpu is installed.",
        "url": "https://heibox.uni-heidelberg.de/f/c2d167ad7d1b486cac48/?dl=1",
    },
    "cpu": {
        "description": "Cpu only version, smaller installation size.",
        "url": "https://heibox.uni-heidelberg.de/f/758d076cfadc46a89980/?dl=1",
    },
}


def download_plantseg(out_path):
    print("Please choose the plantseg version, options available are are:")
    for key, value in plantseg_win_urls.items():
        print(f"   {key}: {value['description']}")

    options = "/".join(plantseg_win_urls.keys())
    key = input(f"please type one option between {options}: ")
    if key in plantseg_win_urls.keys():
        url = plantseg_win_urls[key]["url"]

        print(" -Downloading plantseg... ")
        with requests.get(f"{url}", allow_redirects=True) as r:
            with open(out_path, "wb") as f:
                f.write(r.content)
    else:
        raise ValueError(
            f'Key "{key}" is not a valid option, please select one between {options}'
        )


def unzip_plantseg(zip_path, out_path):
    print(" -Unzipping plantseg... ")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(out_path)


def cleanup(zip_path):
    print(" -Clean up... ")
    os.remove(zip_path)


def modify_config(base_path, file):
    config_path = os.path.join(
        base_path,
        "Lib",
        "site-packages",
        "plantseg",
        "resources",
        "config_gui_template.yaml",
    )

    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # edit config with file
        config["path"] = file

        with open(config_path, "w") as outfile:
            yaml.dump(config, outfile)


def check():
    print("Check PlantSeg environment \n")
    home_path = os.path.expanduser("~")
    configs_path = os.path.join(home_path, PLANTSEG_MODELS_DIR)
    os.makedirs(configs_path, exist_ok=True)

    plantseg_zip_path = os.path.join(configs_path, PLANTSEG_BIN_ZIP)
    plantseg_dir_path = os.path.join(configs_path, PLANTSEG_BIN_DIR)

    if not os.path.isdir(plantseg_dir_path):
        print(
            f"plantseg Installation... \n"
            f" plantseg is going to be installed in {configs_path},"
            f" to uninstall plantseg you can simply delete this directory."
        )
        install_permission = input(
            "Do you want to proceed with the Installation? (y/n)"
        )
        if install_permission != "y":
            exit(0)

        download_plantseg(plantseg_zip_path)
        unzip_plantseg(plantseg_zip_path, plantseg_dir_path)
        cleanup(plantseg_zip_path)
    else:
        print(f"plantseg installation found in {plantseg_dir_path}")

    return plantseg_dir_path


def run(plantseg_bin_dir):
    python_path = os.path.join(plantseg_bin_dir, "python.exe")
    run_script_path = os.path.join(
        plantseg_bin_dir, "Lib", "site-packages", "plantseg", "run_plantseg.py"
    )

    args = sys.argv
    if len(args) == 1:
        subprocess.run([python_path, run_script_path, "--napari"])

    elif len(args) == 2:
        file = args[1]
        extension = os.path.splitext(file)[1]
        if extension == ".yaml":
            subprocess.run([python_path, run_script_path, "--config", file])

        if extension == ".pkl":
            subprocess.run([python_path, run_script_path, "--headless", file])

        elif extension in [".h5", ".hdf", ".tiff", ".tif"] or os.path.isdir(file):
            modify_config(plantseg_bin_dir, file)
            subprocess.run([python_path, run_script_path, "--gui"])


if __name__ == "__main__":
    plantseg_path = check()
    run(plantseg_path)
