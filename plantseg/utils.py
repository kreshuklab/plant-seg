import importlib
from shutil import rmtree
from pathlib import Path
from warnings import warn
from concurrent.futures import ThreadPoolExecutor

import requests
import yaml

from plantseg import PATH_PLANTSEG_MODELS
from plantseg.pipeline import gui_logger


def load_config(config_path: Path) -> dict:
    """Load a YAML configuration file into a dictionary."""
    config_path = Path(config_path)
    with config_path.open('r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def save_config(config: dict, config_path: Path) -> None:
    """Save a dictionary to a YAML configuration file."""
    config_path = Path(config_path)
    with config_path.open('w') as file:
        yaml.dump(config, file)


def download_file(url: str, filename: Path) -> None:
    """Download a single file from a URL to a specified filename."""
    try:
        response = requests.get(url, stream=True)  # Use stream for large files
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):  # Adjust chunk size as needed
                f.write(chunk)
    except requests.RequestException as e:
        warn(f"Failed to download {url}. Error: {e}")


def download_files(urls: dict, out_dir: Path) -> None:
    """Download files from URLs to a specified directory."""
    out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)  # Create the directory and any parent directories

    with ThreadPoolExecutor() as executor:
        futures = []
        for filename, url in urls.items():
            file_path = out_dir / filename
            if not file_path.exists():  # Skip download if file already exists
                gui_logger.info(f"Downloading file {filename} from {url}...")
                futures.append(executor.submit(download_file, url, file_path))
            else:
                gui_logger.info(f"File {filename} already exists. Skipping download.")

        for future in futures:
            future.result()  # Wait for all downloads to complete


def clean_models() -> None:
    """Delete all models in the model zoo after confirmation from the user."""
    for _ in range(3):
        answer = input("This will delete all models in the model zoo."
                       "Ensure you've backed up custom models. Continue? (y/n): ").strip().lower()
        if answer == 'y':
            try:
                rmtree(PATH_PLANTSEG_MODELS, ignore_errors=True)
                print("All models deleted successfully.")
            except Exception as e:
                print(f"An error occurred while trying to delete models: {e}")
            finally:
                print("Operation complete. PlantSeg will now close.")
                break
        elif answer == 'n':
            print("Operation cancelled. No models were deleted.")
            break
        else:
            print("Invalid input, please type 'y' or 'n'.")


def check_version(current_version: str, plantseg_url: str = 'https://api.github.com/repos/kreshuklab/plant-seg/releases/latest') -> None:
    """Check for the latest version of PlantSeg available on GitHub."""
    try:
        response = requests.get(plantseg_url)
        response.raise_for_status()  # Raises an HTTPError if the response status code was unsuccessful
        latest_version = response.json()['tag_name']

        # Splitting the version string into components and comparing as tuples of integers
        if tuple(map(int, latest_version.strip('v').split('.'))) > tuple(map(int, current_version.strip('v').split('.'))):
            print(f"New version of PlantSeg available: {latest_version}. Please update to the latest version.")
        else:
            print(f"You are using the latest version of PlantSeg: {current_version}.")
    except requests.RequestException as e:
        warn(f"Could not check for new version. Error: {e}")


def get_class(class_name, modules):
    """Get a class by name from a list of modules."""
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f'Unsupported class: {class_name}')
