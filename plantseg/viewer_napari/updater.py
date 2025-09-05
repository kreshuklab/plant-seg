from napari.qt.threading import thread_worker

from plantseg.viewer_napari import log


def update():
    try:
        import conda.cli.python_api as api
        from conda.exceptions import PackageNotInstalledError
    except ModuleNotFoundError:
        log(
            "Conda not found! Please update manually!",
            thread="updater",
            level="WARNING",
        )
        return

    @thread_worker
    def _update():
        try:
            api.run_command("update", "plant-seg")
        except PackageNotInstalledError:
            log(
                "Unable to update! If you have installed via git, please update your local repo!",
                thread="updater",
                level="WARNING",
            )
            return

        log("Update finished, please restart!", thread="updater", level="INFO")

    log("Starting update, GUI might freeze!", thread="updater", level="INFO")
    _update().start()
