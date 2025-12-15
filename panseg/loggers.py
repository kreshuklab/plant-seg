import logging
import sys

formatter_viewer_napari = logging.Formatter(
    "%(message)s"
)  # Napari prepends '%(levelname)s: '.upper()
formatter_default = logging.Formatter(
    "%(levelname)s: %(handler_name)s [%(threadName)s] %(asctime)s %(name)s - %(message)s"
)


class PanSegHandler(logging.StreamHandler):
    """A handler that adds handler_name records when emitting log records.

    Please `.set_name("HANDLER_NAME")` before adding to a logger.
    """

    def emit(self, record):
        record.handler_name = self.get_name()
        super().emit(record)


stream_handler = PanSegHandler(sys.stdout)
stream_handler.set_name("P")
stream_handler.setFormatter(formatter_default)
