import importlib
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from shutil import rmtree

import requests
import yaml
from packaging.version import Version

from plantseg import PATH_PLANTSEG_MODELS

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load a YAML configuration file into a dictionary."""
    config_path = Path(config_path)
    with config_path.open("r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def save_config(config: dict, config_path: Path) -> None:
    """Save a dictionary to a YAML configuration file."""
    config_path = Path(config_path)
    with config_path.open("w") as file:
        yaml.dump(config, file)


def download_file(url: str, filename: Path) -> None:
    """Download a single file from a URL to a specified filename."""
    try:
        response = requests.get(url, stream=True)  # Use stream for large files
        response.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):  # Adjust chunk size as needed
                f.write(chunk)
    except requests.RequestException as e:
        logger.warning(f"Failed to download {url}. Error: {e}")


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
                logger.info(f"Downloading file {filename} from {url}...")
                futures.append(executor.submit(download_file, url, file_path))
            else:
                logger.info(f"File {filename} already exists. Skipping download.")

        for future in futures:
            future.result()  # Wait for all downloads to complete


def clean_models() -> None:
    """Delete all models in the model zoo after confirmation from the user."""
    for _ in range(3):
        answer = (
            input(
                "This will delete all models in the model zoo.Ensure you've backed up custom models. Continue? (y/n): "
            )
            .strip()
            .lower()
        )
        if answer == "y":
            try:
                rmtree(PATH_PLANTSEG_MODELS, ignore_errors=True)
                logger.info("All models deleted successfully.")
            except Exception as e:
                logger.warning(f"An error occurred while trying to delete models: {e}")
            finally:
                logger.info("Operation complete. PlantSeg will now close.")
                break
        elif answer == "n":
            logger.info("Operation cancelled. No models were deleted.")
            break
        else:
            logger.warning("Invalid input, please type 'y' or 'n'.")


def check_version(
    current_version: str, plantseg_url: str = "https://api.github.com/repos/kreshuklab/plant-seg/releases/latest"
) -> None:
    """
    Check for the latest version of PlantSeg available on GitHub.

    Args:
        current_version (str): The current version of PlantSeg in use.
        plantseg_url (str): The URL to check the latest release of PlantSeg (default is the GitHub API URL).

    Returns:
        None
    """
    try:
        response = requests.get(plantseg_url)
        response.raise_for_status()  # Raises an HTTPError if the response status code was unsuccessful
        latest_version = response.json()["tag_name"]

        current_version_obj = Version(current_version)
        latest_version_obj = Version(latest_version)

        if latest_version_obj > current_version_obj:
            logger.warning(f"New version of PlantSeg available: {latest_version}. Please update to the latest version.")
        else:
            logger.info(f"You are using the latest version of PlantSeg: {current_version}.")
    except requests.RequestException as e:
        logger.warning(f"Could not check for new version. Error: {e}")
    except ValueError as e:
        logger.warning(f"Could not parse version information. Error: {e}")


def get_class(class_name, modules):
    """Get a class by name from a list of modules."""
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f"Unsupported class: {class_name}")
