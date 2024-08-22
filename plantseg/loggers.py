import logging
import sys


# Define the custom formatter to match Napari's format
class PlantSegFormatter(logging.Formatter):
    def __init__(self, mode: str):
        """Custom formatter for PlantSeg logger.

        Args:
            mode (str): a string used to inform users about the mode of PlantSeg (e.g. Napari, headless, etc.)
        """
        super().__init__()
        self.mode = mode

    def format(self, record):
        time_stamp = self.formatTime(record, "%H:%M:%S %d.%m.%Y")
        return f'PlantSeg {self.mode} {record.levelname.lower()} - {time_stamp} - {record.threadName}: {record.getMessage()}'


# Create a PlantSeg logger
gui_logger = logging.getLogger("PlantSeg")
gui_logger.setLevel(logging.INFO)

# Add console handler with the custom formatter
stream_handler = logging.StreamHandler(sys.stdout)
formatter = PlantSegFormatter(mode="Napari")  # TODO: Should change according to the mode
stream_handler.setFormatter(formatter)
gui_logger.addHandler(stream_handler)
