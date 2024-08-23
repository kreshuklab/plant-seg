import logging
import sys

LOGGING_FORMAT = '[%(threadName)s] %(asctime)s %(name)s - %(message)s'
formatter_viewer_napari = logging.Formatter(LOGGING_FORMAT)  # Napari prepends '%(levelname)s: '.upper()
formatter_default = logging.Formatter('%(levelname)s: ' + LOGGING_FORMAT)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter_default)

logger_root = logging.getLogger("PlantSeg")
logger_root.setLevel(logging.INFO)
logger_root.addHandler(stream_handler)

logger_logger = logging.getLogger("PlantSeg.Logger")
