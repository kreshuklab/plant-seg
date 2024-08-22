import logging

from napari.utils.notifications import show_console_notification, show_error, show_info, show_warning

from plantseg.loggers import PlantSegFormatter, gui_logger

napari_notifications = {  # Mapping logging levels to Napari notification functions
    logging.INFO: show_info,
    logging.WARNING: show_warning,
    logging.ERROR: show_error,
    logging.DEBUG: show_console_notification,
}


class NapariHandler(logging.Handler):
    def emit(self, record):
        try:
            message = self.format(record)
            level = record.levelno
            if level in napari_notifications:
                napari_notifications[level](message)
            else:
                show_console_notification(message)
        except Exception:
            self.handleError(record)


# Add the NapariHandler to the PlantSeg logger; TODO: Should be done according to the mode
napari_handler = NapariHandler()
napari_handler.setFormatter(PlantSegFormatter("Napari"))
gui_logger.addHandler(napari_handler)
