# third party
from typer.testing import CliRunner

# first party
from src.cli import app

runner = CliRunner()


def test_hello() -> None:
    result = runner.invoke(app, ["hello"])
    assert result.exit_code == 0
    assert "Syft CLI 👋" in result.stdout
