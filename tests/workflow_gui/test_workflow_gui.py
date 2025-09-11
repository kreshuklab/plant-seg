from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from plantseg.workflow_gui.editor import Workflow_gui


@pytest.fixture
def mock_logger(mocker):
    """Fixture to mock the logger."""
    return mocker.patch("plantseg.workflow_gui.widgets.logger")


@pytest.fixture
def gui(qtbot, workflow_yaml):
    gui = Workflow_gui(config_path=workflow_yaml, run=False)
    qtbot.addWidget(gui.main_window.native)
    return gui


def test_loading_dialog(gui, workflow_yaml):
    with open(workflow_yaml, "r") as f:
        parsed_in = yaml.safe_load(f)
    parsed_in["runner"] = "serial"
    assert parsed_in == gui.config

    gui.change_config.call_button.native.click()
    assert gui.config is None

    gui.loader_w.config_path.value = str(workflow_yaml)
    gui.loader_w.call_button.native.click()
    assert parsed_in == gui.config


def test_load_save_unchanged(gui, workflow_yaml, tmp_path):
    out_file = tmp_path / "test_workflow_out.yaml"
    with open(workflow_yaml, "r") as f:
        parsed_in = yaml.safe_load(f)
    parsed_in["runner"] = "serial"
    assert parsed_in == gui.config

    gui.save_b.native.click()
    gui.save.path.value = str(out_file)
    gui.save.call_button.native.click()

    with open(out_file, "r") as f:
        parsed_out = yaml.safe_load(f)

    # Step through task_list
    for k, v in parsed_in.items():
        if isinstance(v, list):
            for i, l in enumerate(v):
                if isinstance(l, dict):
                    for kk, vv in l.items():
                        assert vv == parsed_out.get(k)[i].get(kk)

        else:
            assert v == parsed_out.get(k)

    assert parsed_out == parsed_in


def test_load_save_unchanged_complete(gui, workflow_complete_yaml, tmp_path):
    out_file = tmp_path / "test_workflow_out.yaml"
    with open(workflow_complete_yaml, "r") as f:
        parsed_in = yaml.safe_load(f)
    parsed_in["runner"] = "serial"
    assert parsed_in == gui.config

    gui.save_b.native.click()
    gui.save.path.value = str(out_file)
    gui.save.call_button.native.click()

    with open(out_file, "r") as f:
        parsed_out = yaml.safe_load(f)

    # Step through task_list
    for k, v in parsed_in.items():
        if isinstance(v, list):
            for i, l in enumerate(v):
                if isinstance(l, dict):
                    for kk, vv in l.items():
                        assert vv == parsed_out.get(k)[i].get(kk)

        else:
            assert v == parsed_out.get(k)

    assert parsed_out == parsed_in


def test_load_missing_config(gui, mock_logger, workflow_yaml):
    gui.change_config.call_button.native.click()

    gui.loader_w.config_path.value = str(workflow_yaml.with_name("NOT_EXISTING.yaml"))
    gui.loader_w.call_button.native.click()
    mock_logger.error.assert_called_with("File does not exist!")

    gui.loader_w.config_path.value = str(
        Path(__file__).resolve().parent.parent / "resources" / "rgb_3D.tif"
    )
    gui.loader_w.call_button.native.click()
    mock_logger.error.assert_called_with("Please provide a yaml file!")


def test_invalid_config(gui):
    with patch("plantseg.workflow_gui.editor.logger") as mock_logger:
        conf: dict = deepcopy(gui.config)

        conf_mod = deepcopy(conf)
        conf_mod.pop("list_tasks")
        gui.config = conf_mod
        gui.show_config()
        mock_logger.error.assert_called_once()
        mock_logger.reset_mock()

        conf_mod = deepcopy(conf)
        conf_mod.pop("infos")
        gui.config = conf_mod
        gui.show_config()
        mock_logger.error.assert_called_once()
        mock_logger.reset_mock()

        conf_mod = deepcopy(conf)
        conf_mod.pop("inputs")
        gui.config = conf_mod
        gui.show_config()
        mock_logger.error.assert_called_once()
        mock_logger.reset_mock()

        gui.config = conf["list_tasks"]
        gui.show_config()
        mock_logger.error.assert_called_once()
        mock_logger.reset_mock()


def test_toggle_theme(gui):
    gui.toggle_theme()
    gui.toggle_theme()
    gui.toggle_theme()
