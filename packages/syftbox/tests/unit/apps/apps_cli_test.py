import pytest
from typer.testing import CliRunner

from syftbox.app.cli import app as app_cli
from syftbox.app.install import InstallResult

# Initialize test runner
runner = CliRunner()


@pytest.fixture
def mock_apps_dir(mock_config):
    apps_dir = mock_config.data_dir / "apis"
    apps_dir.mkdir(exist_ok=True)
    yield apps_dir


@pytest.fixture
def mock_app_list(mock_apps_dir):
    apps = [mock_apps_dir / "app1", mock_apps_dir / "app2"]
    for app in apps:
        app.mkdir(exist_ok=True)
    yield (mock_apps_dir, apps)


@pytest.fixture
def mocked_app_install(monkeypatch, mock_apps_dir):
    def mock_install(config, repository, branch):
        return InstallResult(
            app_name=repository,
            app_path=mock_apps_dir / repository,
            error=None,
            details=None,
        )

    monkeypatch.setattr("syftbox.app.cli.install_app", mock_install)
    yield
    monkeypatch.undo()


@pytest.fixture
def mocked_app_uninstall(monkeypatch, mock_config):
    def mock_uninstall(*args, **kwargs):
        return mock_config.data_dir / "apps"

    monkeypatch.setattr("syftbox.app.cli.uninstall_app", mock_uninstall)
    yield
    monkeypatch.undo()


def test_list_app(mock_app_list):
    result = runner.invoke(app_cli, ["list"])
    assert result.exit_code == 0
    assert "Apps installed" in result.stdout
    assert "app1" in result.stdout
    assert "app2" in result.stdout


def test_list_app_empty(mock_apps_dir):
    result = runner.invoke(app_cli, ["list"])
    assert result.exit_code == 0
    assert "No apps installed" in result.stdout


def test_install_app(mocked_app_install):
    result = runner.invoke(app_cli, ["install", "OpenMined/tutorials-app"])
    assert result.exit_code == 0
    assert "tutorials-app" in result.stdout


def test_uninstall_app(mocked_app_uninstall):
    result = runner.invoke(app_cli, ["uninstall", "app1"])
    assert result.exit_code == 0
    assert "Uninstalled app" in result.stdout
    assert "app1" in result.stdout
