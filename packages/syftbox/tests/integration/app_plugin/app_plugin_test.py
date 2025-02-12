import threading
import time
from pathlib import Path
from secrets import token_hex

from syftbox.client.plugins.apps import run_apps
from tests.integration.app_plugin.fixtures.app_mocks import AppMockFactory


def verify_app_execution(app_dir: Path, expected_output: str):
    """Verify app execution results."""
    app_log_path = app_dir / "app.log"
    assert app_log_path.exists()
    assert app_log_path.read_text().strip() == expected_output


def verify_running_apps(running_apps: dict, expected_app_name: str = None):
    """Verify running apps state."""
    if expected_app_name:
        assert len(running_apps) == 1
        assert expected_app_name in running_apps
    else:
        assert len(running_apps) == 0


def test_app_plugin_without_config(tmp_path, monkeypatch):
    """Test app plugin execution without configuration."""
    app_dir = tmp_path / token_hex(8)
    app_name = "test_app_without_config"
    mock_app_dir, expected_output = AppMockFactory.create_app_without_config(apps_dir=app_dir, app_name=app_name)

    assert mock_app_dir.exists()

    # Patch necessary attributes
    PATCHED_RUNNING = {}
    monkeypatch.setattr("syftbox.client.plugins.apps.DEFAULT_APPS_PATH", "")
    monkeypatch.setattr("syftbox.client.plugins.apps.RUNNING_APPS", PATCHED_RUNNING)

    # Run app
    no_config = ""  # dummy app doesn't need SYFTBOX_CLIENT_CONFIG_PATH
    run_apps(app_dir, no_config)

    # Verify results
    verify_running_apps(PATCHED_RUNNING)
    verify_app_execution(mock_app_dir, expected_output)


def test_app_plugin_with_config(tmp_path, monkeypatch):
    """Test app plugin execution with configuration."""
    app_dir = tmp_path / token_hex(8)
    app_name = "test_app_with_config"
    mock_app_dir, expected_output = AppMockFactory.create_app_with_config(apps_dir=app_dir, app_name=app_name)

    assert mock_app_dir.exists()

    # Patch necessary attributes
    PATCHED_RUNNING = {}
    EVT = threading.Event()
    monkeypatch.setattr("syftbox.client.plugins.apps.DEFAULT_APPS_PATH", "")
    monkeypatch.setattr("syftbox.client.plugins.apps.RUNNING_APPS", PATCHED_RUNNING)
    monkeypatch.setattr("syftbox.client.plugins.apps.EVENT", EVT)

    # Run app
    no_config = ""  # dummy app doesn't need SYFTBOX_CLIENT_CONFIG_PATH
    run_apps(app_dir, no_config)
    time.sleep(2)

    # Verify results
    verify_running_apps(PATCHED_RUNNING, app_name)
    verify_app_execution(mock_app_dir, expected_output)

    # This doesn't kill the process gracefully,
    # later need to implement a graceful shutdown mechanism for apps
    if app_name in PATCHED_RUNNING:
        EVT.set()
        app_thread: threading.Thread = PATCHED_RUNNING[app_name]
        app_thread.join(timeout=1)
