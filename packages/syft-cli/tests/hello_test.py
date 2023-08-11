# third party
from cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_hello():
    result = runner.invoke(app, ["hello"])
    assert result.exit_code == 0
    assert "Syft CLI 👋" in result.stdout
