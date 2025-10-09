from napari.qt.threading import thread_worker

from plantseg.viewer_napari import log
import subprocess


def update():
    @thread_worker
    def _update():
        try:
            subprocess.run(
                ["conda", "update", "plant-seg"],
                input="y\n",
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            log(
                "Unable to update! If you have installed via git, please update your local repo!",
                thread="updater",
                level="WARNING",
            )
            return

        log("Update finished, please restart!", thread="updater", level="INFO")

    log(
        "Starting update, might take a while!\nCheck the progress in the terminal.",
        thread="updater",
        level="INFO",
    )
    _update().start()
