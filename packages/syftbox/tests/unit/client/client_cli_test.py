from pathlib import Path

from typer.testing import CliRunner

from syftbox.client.cli import app as client_cli

# Initialize test runner
runner = CliRunner()


def mock_port_in_use(*args, **kwargs):
    return False


def test_run_syftbox_success(monkeypatch, mock_config):
    def setup_config_interactive(*args, **kwargs):
        return mock_config

    def mock_run_syftbox(*args, **kwargs):
        return 0

    def get_migration_decision(*args, **kwargs):
        return False

    monkeypatch.setattr("syftbox.client.core.run_syftbox", mock_run_syftbox)
    monkeypatch.setattr("syftbox.client.cli_setup.setup_config_interactive", setup_config_interactive)
    monkeypatch.setattr("syftbox.client.cli_setup.get_migration_decision", get_migration_decision)
    monkeypatch.setattr("syftbox.client.utils.net.is_port_in_use", mock_port_in_use)

    result = runner.invoke(client_cli)
    assert result.exit_code == 0


def test_client_error(monkeypatch, mock_config):
    def setup_config_interactive(*args, **kwargs):
        return mock_config

    def mock_run_syftbox(*args, **kwargs):
        return -1

    def get_migration_decision(*args, **kwargs):
        return False

    monkeypatch.setattr("syftbox.client.core.run_syftbox", mock_run_syftbox)
    monkeypatch.setattr("syftbox.client.cli_setup.setup_config_interactive", setup_config_interactive)
    monkeypatch.setattr("syftbox.client.utils.net.is_port_in_use", mock_port_in_use)
    monkeypatch.setattr("syftbox.client.cli_setup.get_migration_decision", get_migration_decision)

    result = runner.invoke(client_cli)
    assert result.exit_code == -1


def test_port_error(monkeypatch):
    monkeypatch.setattr("syftbox.client.utils.net.is_port_in_use", lambda p: True)
    result = runner.invoke(client_cli)
    assert result.exit_code == 1


def test_client_report(monkeypatch, tmp_path, mock_config):
    monkeypatch.setattr("syftbox.client.logger.zip_logs", lambda p, **kw: Path(str(p) + ".log"))
    result = runner.invoke(client_cli, ["report"])
    assert result.exit_code == 0


def test_client_benchmark(monkeypatch, tmp_path, mock_config):
    def mock_generate(*args, **kwargs):
        return

    def mock_run(*args, **kwargs):
        return

    monkeypatch.setattr("syftbox.client.benchmark.runner.BenchmarkReporter.generate", mock_generate)
    monkeypatch.setattr("syftbox.client.benchmark.runner.SyftBenchmarkRunner.run", mock_run)

    result = runner.invoke(client_cli, ["benchmark"])
    assert result.exit_code == 0
