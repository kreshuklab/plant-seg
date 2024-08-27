import logging

from plantseg.viewer_napari.loggers import napari_handler

logger_viewer_napari = logging.getLogger(__name__)
logger_viewer_napari.setLevel(logging.INFO)
logger_viewer_napari.addHandler(napari_handler)

# Avoid propagating to loggers.stream_handler
logger_viewer_napari.propagate = False
logger_viewer_napari.info("Napari logger configured. Napari logger name: {__name__}")


def napari_formatted_logging(message: str, thread: str, level: str = 'info'):
    """Deprecated function for logging into Napari GUI."""
    logger_viewer_napari.log(logging.getLevelName(level.upper()), message)
