import logging
import sys

LOGGING_FORMAT = '%(handler_name)s [%(threadName)s] %(asctime)s %(name)s - %(message)s'
formatter_viewer_napari = logging.Formatter(LOGGING_FORMAT)  # Napari prepends '%(levelname)s: '.upper()
formatter_default = logging.Formatter('%(levelname)s: ' + LOGGING_FORMAT)


class PlantSegHandler(logging.StreamHandler):
    """A handler that adds handler_name records when emitting log records.

    Please `.set_name("HANDLER_NAME")` before adding to a logger.
    """

    def emit(self, record):
        record.handler_name = self.get_name()
        super().emit(record)


stream_handler = PlantSegHandler(sys.stdout)
stream_handler.set_name("P")
stream_handler.setFormatter(formatter_default)

logger_root = logging.getLogger("PlantSeg")
logger_root.setLevel(logging.INFO)
logger_root.addHandler(stream_handler)

logger_logger = logging.getLogger("PlantSeg.Logger")
