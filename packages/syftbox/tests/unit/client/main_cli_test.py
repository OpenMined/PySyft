from typer.testing import CliRunner

from syftbox.main import app as main_cli

# Initialize test runner
runner = CliRunner()


def test_version():
    result = runner.invoke(main_cli, ["version"])
    assert result.exit_code == 0
    assert len(result.stdout.split(".")) == 3


def test_debug(monkeypatch):
    def mock_debug_report(*args, **kwargs):
        return "ok: true"

    monkeypatch.setattr("syftbox.lib.debug.debug_report_yaml", mock_debug_report)
    result = runner.invoke(main_cli, ["debug"])
    assert result.exit_code == 0, result.stdout
    assert "ok: true" in result.stdout


def test_debug_invalid_config(monkeypatch):
    def mock_debug_report(*args, **kwargs):
        raise Exception("Invalid config")

    monkeypatch.setattr("syftbox.lib.debug.debug_report_yaml", mock_debug_report)

    result = runner.invoke(main_cli, ["debug"])
    assert result.exit_code == 1
    assert "Error" in result.stdout


def test_debug_custom_config(monkeypatch, mock_config):
    def mock_debug_report(conf_path):
        assert str(conf_path) == str(mock_config.path)
        return "ok: true"

    monkeypatch.setattr("syftbox.lib.debug.debug_report_yaml", mock_debug_report)

    result = runner.invoke(main_cli, ["debug", "--config", str(mock_config.path)])
    assert result.exit_code == 0
    assert "ok: true" in result.stdout
