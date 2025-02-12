from loguru import logger
from typer import Exit, Option, Typer

from syftbox.server.migrations import run_migrations
from syftbox.server.settings import ServerSettings

app = Typer(
    name="SyftBox Server",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


# Define options separately to keep the function signature clean
# fmt: off
SERVER_PANEL = "Server Options"
SSL_PANEL = "SSL Options"

EXAMPLE_OPTS = Option(
    "-v", "--verbose",
    is_flag=True,
    rich_help_panel=SERVER_PANEL,
    help="Enable verbose mode",
)
# fmt: on


@app.command()
def migrate() -> None:
    """Run database migrations"""

    try:
        settings = ServerSettings()
        run_migrations(settings)
        logger.info("Migrations completed successfully")
    except Exception as e:
        logger.error("Migrations failed")
        logger.exception(e)
        raise Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
