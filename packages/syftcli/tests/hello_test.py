# third party
from typer.testing import CliRunner

# syftcli absolute
from syftcli.cli import app

runner = CliRunner()


def test_hello() -> None:
    result = runner.invoke(app, ["hello"])
    assert result.exit_code == 0
    assert "Syft CLI 👋" in result.stdout
