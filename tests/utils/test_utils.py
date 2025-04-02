import pytest
import requests

from plantseg.utils import check_version


@pytest.fixture
def mock_logger(mocker):
    """Fixture to mock the logger."""
    return mocker.patch("plantseg.utils.logger")


def test_check_version_new_version(mock_logger, requests_mock):
    """Test when the latest version is newer than the current version."""
    current_version = "1.0.0"
    latest_version = "2.0.0"

    # Mock the API response
    requests_mock.get(
        "https://api.github.com/repos/kreshuklab/plant-seg/releases/latest",
        json={"tag_name": latest_version},
    )

    check_version(current_version)

    # Assert logger warning was called with appropriate message
    mock_logger.warning.assert_called_once_with(
        f"New version of PlantSeg available: {latest_version}. Please update to the latest version."
    )


def test_check_version_same_version(mock_logger, requests_mock):
    """Test when the current version is the same as the latest version."""
    current_version = "2.0.0"
    latest_version = "2.0.0"

    # Mock the API response
    requests_mock.get(
        "https://api.github.com/repos/kreshuklab/plant-seg/releases/latest",
        json={"tag_name": latest_version},
    )

    check_version(current_version)

    # Assert logger info was called with appropriate message
    mock_logger.info.assert_called_once_with(
        f"You are using the latest version of PlantSeg: {current_version}."
    )


def test_check_version_old_version(mock_logger, requests_mock):
    """Test when the current version is older than the latest version."""
    current_version = "2.0.0"
    latest_version = "1.9.0"

    # Mock the API response
    requests_mock.get(
        "https://api.github.com/repos/kreshuklab/plant-seg/releases/latest",
        json={"tag_name": latest_version},
    )

    check_version(current_version)

    # Assert logger info was called with appropriate message
    mock_logger.info.assert_called_once_with(
        f"You are using the latest version of PlantSeg: {current_version}."
    )


def test_check_version_beta_version(mock_logger, requests_mock):
    """Test when the latest version is a beta version."""
    current_version = "2.0.0"
    latest_version = "2.0.0b3"

    # Mock the API response
    requests_mock.get(
        "https://api.github.com/repos/kreshuklab/plant-seg/releases/latest",
        json={"tag_name": latest_version},
    )

    check_version(current_version)

    # Assert logger info was called with appropriate message
    mock_logger.info.assert_called_once_with(
        f"You are using the latest version of PlantSeg: {current_version}."
    )


def test_check_version_new_beta_version(mock_logger, requests_mock):
    """Test when the current version is older and the latest version is a beta."""
    current_version = "1.0.0"
    latest_version = "2.0.0b3"

    # Mock the API response
    requests_mock.get(
        "https://api.github.com/repos/kreshuklab/plant-seg/releases/latest",
        json={"tag_name": latest_version},
    )

    check_version(current_version)

    # Assert logger warning was called with appropriate message
    mock_logger.warning.assert_called_once_with(
        f"New version of PlantSeg available: {latest_version}. Please update to the latest version."
    )


def test_check_version_request_exception(mock_logger, requests_mock):
    """Test when the GitHub API request fails."""
    current_version = "1.0.0"

    # Mock the API to raise a RequestException
    requests_mock.get(
        "https://api.github.com/repos/kreshuklab/plant-seg/releases/latest",
        exc=requests.RequestException,
    )

    check_version(current_version)

    # Assert logger warning was called with appropriate message
    mock_logger.warning.assert_called_once_with(
        "Could not check for new version. Error: "
    )


def test_check_version_value_error(mock_logger, requests_mock):
    """Test when the version format is invalid."""
    current_version = "1.0.0"

    # Mock the API response with an invalid version format
    requests_mock.get(
        "https://api.github.com/repos/kreshuklab/plant-seg/releases/latest",
        json={"tag_name": "invalid_version"},
    )

    check_version(current_version)

    # Assert logger warning was called with appropriate message
    mock_logger.warning.assert_called_once_with(
        "Could not parse version information. Error: Invalid version: 'invalid_version'"
    )
