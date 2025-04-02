import logging

from napari.utils.notifications import (
    show_console_notification,
    show_error,
    show_info,
    show_warning,
)

from plantseg.loggers import formatter_viewer_napari

napari_notifications = {  # Mapping logging levels to Napari notification functions
    logging.INFO: show_info,
    logging.WARNING: show_warning,
    logging.ERROR: show_error,
    logging.DEBUG: show_console_notification,
}


class NapariHandler(logging.Handler):
    """Custom logging handler for logging into Napari GUI with default logging API.

    To show widget name in the log message.
    """

    def emit(self, record):
        try:
            record.handler_name = self.get_name()
            assert hasattr(record, "widget_name"), (
                "For Napari logging, use 'napari_formatted_logging' in log record."
            )
            message = self.format(record)
            level = record.levelno
            napari_notifications[level](message)
        except Exception:
            self.handleError(record)


# Set up logging from Napari
napari_handler = NapariHandler()
napari_handler.set_name("N")
napari_handler.setFormatter(formatter_viewer_napari)
