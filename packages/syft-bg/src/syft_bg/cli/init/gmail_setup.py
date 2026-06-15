"""Gmail OAuth setup for notification service."""

from pathlib import Path

import click

from syft_bg.cli.init.exceptions import InitFlowError
from syft_bg.notify.gmail import GmailAuth


def setup_gmail(
    credentials_path: Path,
    result_token_path: Path,
    skip: bool = False,
    quiet: bool = False,
) -> None:
    """Set up Gmail authentication.

    Args:
        credentials_path: Path to OAuth credentials.json
        token_path: Path to save gmail_token.json
        skip: If True, skip OAuth setup (token must already exist)
        quiet: If True, suppress output messages

    Raises:
        InitFlowError: If setup fails.
    """
    if result_token_path.exists():
        if not quiet:
            click.echo(f"Gmail token exists: {result_token_path}")
        return

    if skip:
        raise InitFlowError(
            f"Cannot skip Gmail OAuth - token not found\n"
            f"  Expected: {result_token_path}\n\n"
            f"Either:\n"
            f"  - Run without --skip-oauth to complete Gmail authentication\n"
            f"  - Provide existing token with --gmail-token /path/to/token.json"
        )

    if not quiet:
        click.echo("Gmail is required for email notifications.")
        click.echo()

    if not credentials_path.exists():
        click.echo(f"credentials.json not found at {credentials_path}")
        click.echo()
        click.echo("To get credentials.json:")
        click.echo("  1. Go to Google Cloud Console -> APIs & Services -> Credentials")
        click.echo("  2. Create OAuth 2.0 Client ID (Desktop app)")
        click.echo("  3. Download as credentials.json")
        click.echo(f"  4. Place it at: {credentials_path}")
        click.echo()

        if quiet:
            raise InitFlowError(f"credentials.json not found at {credentials_path}")

        creds_input = click.prompt(
            "Or enter path to credentials.json", type=click.Path(exists=True)
        )
        credentials_path = Path(creds_input).expanduser()

    if not quiet:
        click.echo("Setting up Gmail authentication...")

    try:
        gmail_auth = GmailAuth()
        user_tokens = gmail_auth.authenticate_user(credentials_path)
        result_token_path.parent.mkdir(parents=True, exist_ok=True)
        result_token_path.write_text(user_tokens.to_json())
        if not quiet:
            click.echo(f"Gmail token saved: {result_token_path}")
    except ImportError:
        raise InitFlowError("Gmail auth module not available")
    except Exception as e:
        raise InitFlowError(f"Gmail setup failed: {e}") from e
