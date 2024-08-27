import logging

from napari.utils.notifications import show_console_notification, show_error, show_info, show_warning

from plantseg.loggers import formatter_viewer_napari

napari_notifications = {  # Mapping logging levels to Napari notification functions
    logging.INFO: show_info,
    logging.WARNING: show_warning,
    logging.ERROR: show_error,
    logging.DEBUG: show_console_notification,
}


class NapariHandler(logging.Handler):
    """Custom logging handler for logging into Napari GUI with default logging API.

    i.e.
    use `logging.getLogger("PlantSeg.Napari").info("message")`,
    instead of `napari_formatted_logging("message", "thread")` from PlantSeg V1,
    or napari.utils.notifications.show_info("message") from Napari.
    """

    def emit(self, record):
        try:
            record.handler_name = self.get_name()
            message = self.format(record)
            level = record.levelno
            napari_notifications[level](message)
        except Exception:
            self.handleError(record)


def napari_formatted_logging(message: str, thread: str, level: str = 'info'):
    """Deprecated function for logging into Napari GUI."""
    logger_viewer_napari.log(logging.getLevelName(level.upper()), message)


# Set up logging from Napari
napari_handler = NapariHandler()
napari_handler.set_name("N")
napari_handler.setFormatter(formatter_viewer_napari)

logger_viewer_napari = logging.getLogger("PlantSeg.Napari")
logger_viewer_napari.setLevel(logging.INFO)
logger_viewer_napari.addHandler(napari_handler)

# Avoid propagating to loggers.stream_handler
logger_viewer_napari.propagate = False
