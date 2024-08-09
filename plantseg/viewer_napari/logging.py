from datetime import datetime

from napari.utils.notifications import show_info, show_warning, show_error, show_console_notification

napari_notifications = {
    'info': show_info,
    'warning': show_warning,
    'error': show_error,
    'console': show_console_notification,
}


def napari_formatted_logging(message: str, thread: str, level: str = 'info'):
    """Log a message to the napari console with a standardized format.

    Args:
        message (str): Message to log.
        thread (str): Name of the thread that is logging the message.
        level (str): Level of the message. Defaults to 'info'.
    """
    assert level in napari_notifications.keys(), (
        f'Invalid notification type: {level}, ' f'valid types are: {napari_notifications.keys()}'
    )
    time_stamp = datetime.now().strftime("%H:%M:%S %d.%m.%Y")
    formatted_msg = f' PlantSeg Napari {level} - {time_stamp} - {thread}: {message}'
    napari_notifications[level](formatted_msg)
