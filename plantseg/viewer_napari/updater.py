from napari.qt.threading import thread_worker
from plantseg.viewer_napari import log


def update():
    try:
        import conda.cli.python_api as api
    except ModuleNotFoundError:
        log(
            "Conda not found! Please update manually!",
            thread="updater",
            level="WARNING",
        )
        return

    @thread_worker
    def _update():
        api.run_command("update", "plant-seg")

    log("Starting update", thread="updater", level="INFO")
    _update()
    log("Update finished, please restart!", thread="updater", level="INFO")
