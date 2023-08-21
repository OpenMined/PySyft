# third party
from cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "CLI" in result.stdout
    assert "Library" in result.stdout
