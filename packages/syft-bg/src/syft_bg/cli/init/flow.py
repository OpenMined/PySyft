"""Main initialization flow for syft-bg services."""

from dataclasses import dataclass
from pathlib import Path

import click

from syft_bg.cli.init.exceptions import InitFlowError
from syft_bg.common.config import get_default_paths
from syft_bg.common.syft_bg_config import SyftBgConfig


@dataclass
class UserPassedConfig:
    """Configuration parameters for syft-bg init flow.

    All parameters are optional. When None, the init flow will either
    prompt the user (interactive mode) or use defaults (quiet mode).
    """

    # Core settings
    email: str | None = None
    syftbox_root: str | None = None

    # Control flags
    yes: bool = False  # Auto-confirm config update
    quiet: bool = False  # Use defaults, no prompts (implies skip_oauth)
    skip_oauth: bool = False  # Skip OAuth setup (tokens must exist)

    # Notification settings (None = prompt or use default)
    notify_jobs: bool | None = None
    notify_peers: bool | None = None
    notify_interval: int | None = None

    # Job approval settings
    approve_jobs: bool | None = None

    # Peer approval settings
    approve_peers: bool | None = None
    approved_domains: list[str] | None = None
    approve_interval: int | None = None

    # OAuth/credentials paths
    credentials_path: str | None = None
    gmail_token_path: str | None = None
    drive_token_path: str | None = None


def _print_banner(quiet: bool = False) -> None:
    """Print the setup banner."""
    if quiet:
        return
    click.echo()
    click.echo("SYFTBOX BACKGROUND SERVICES SETUP")
    click.echo("=" * 50)
    click.echo()
    click.echo("This will configure both notification and auto-approval services.")
    click.echo()


def _print_section(title: str, quiet: bool = False) -> None:
    """Print a section header."""
    if quiet:
        return
    click.echo()
    click.echo("-" * 50)
    click.echo(title)
    click.echo("-" * 50)
    click.echo()


def _load_existing_config(config: UserPassedConfig, config_path: Path) -> SyftBgConfig:
    """Load existing config and confirm overwrite."""
    if not config_path.exists():
        return SyftBgConfig()

    if not config.quiet:
        click.echo(f"Found existing config at {config_path}")

    return SyftBgConfig.from_path(config_path)


def _resolve_common_settings(config: UserPassedConfig, result: SyftBgConfig) -> None:
    """Resolve email and syftbox_root, mutating result."""
    default_email = result.do_email or ""

    if config.email is not None:
        result.do_email = config.email
        if not config.quiet:
            click.echo(f"Data Owner email address: {result.do_email}")
    elif config.quiet:
        if not default_email:
            raise InitFlowError(
                "--email is required in quiet mode\n"
                "Usage: syft-bg init --email user@example.com --quiet"
            )
        result.do_email = default_email
    else:
        result.do_email = click.prompt(
            "Data Owner email address", default=default_email or None
        )

    from syft_client.sync.syftbox_manager import get_jupyter_default_syftbox_folder

    default_syftbox = result.syftbox_root or str(
        get_jupyter_default_syftbox_folder(result.do_email)
    )
    if config.syftbox_root is not None:
        result.syftbox_root = config.syftbox_root
        if not config.quiet:
            click.echo(f"SyftBox root directory: {result.syftbox_root}")
    elif config.quiet:
        result.syftbox_root = default_syftbox
    else:
        result.syftbox_root = click.prompt(
            "SyftBox root directory", default=default_syftbox
        )


# def _setup_auth(
#     user_passed_config: UserPassedConfig,
#     result_config: SyftBgConfig,
# ) -> None:
#     """Run Gmail and Drive OAuth setup, storing paths on result_config."""
#     result_config.credentials_path = Path(
#         user_passed_config.credentials_path or result_config.credentials_path
#     )

#     result_config.gmail_token_path = Path(
#         user_passed_config.gmail_token_path or result_config.gmail_token_path
#     )
#     result_config.drive_token_path = Path(
#         user_passed_config.drive_token_path or result_config.drive_token_path
#     )

#     _print_section("GOOGLE DRIVE AUTHENTICATION", quiet=user_passed_config.quiet)

#     if is_colab():
#         if not user_passed_config.quiet:
#             click.echo(
#                 "Colab detected - Drive and Gmail authentication handled by Colab"
#             )
#     else:
#         setup_gmail(
#             result_config.credentials_path,
#             result_config.gmail_token_path,
#             skip=user_passed_config.skip_oauth,
#             quiet=user_passed_config.quiet,
#         )

#         setup_drive(
#             result_config.credentials_path,
#             result_config.drive_token_path,
#             skip=user_passed_config.skip_oauth,
#             quiet=user_passed_config.quiet,
#         )


def _resolve_notify_settings(config: UserPassedConfig, result: SyftBgConfig) -> None:
    """Resolve notification service settings, mutating result.notify."""
    notify = result.notify

    if config.notify_jobs is not None:
        notify.monitor_jobs = config.notify_jobs
    elif not config.quiet:
        notify.monitor_jobs = click.confirm(
            "Enable email notifications for new jobs?",
            default=notify.monitor_jobs,
        )

    if config.notify_peers is not None:
        notify.monitor_peers = config.notify_peers
    elif not config.quiet:
        notify.monitor_peers = click.confirm(
            "Enable email notifications for peer requests?",
            default=notify.monitor_peers,
        )

    if config.notify_interval is not None:
        notify.interval = config.notify_interval
    elif not config.quiet:
        notify.interval = click.prompt(
            "Check interval (seconds)",
            type=int,
            default=notify.interval,
        )


def _resolve_approve_settings(config: UserPassedConfig, result: SyftBgConfig) -> None:
    """Resolve auto-approval service settings, mutating result.approve."""
    approve = result.approve

    if not config.quiet:
        click.echo("Job Auto-Approval:")

    if config.approve_jobs is not None:
        approve.auto_approvals.enabled = config.approve_jobs
    elif not config.quiet:
        approve.auto_approvals.enabled = click.confirm(
            "  Enable automatic job approval?",
            default=approve.auto_approvals.enabled,
        )

    if not config.quiet:
        click.echo()
        click.echo("Peer Auto-Approval:")

    if config.approve_peers is not None:
        approve.peers.enabled = config.approve_peers
    elif not config.quiet:
        approve.peers.enabled = click.confirm(
            "  Enable automatic peer approval?",
            default=approve.peers.enabled,
        )

    if approve.peers.enabled:
        if config.approved_domains is not None:
            approve.peers.approved_domains = config.approved_domains
        elif not config.quiet:
            default_domains = ",".join(
                approve.peers.approved_domains or ["openmined.org"]
            )
            domains_input = click.prompt(
                "  Approved domains (comma-separated)", default=default_domains
            )
            approve.peers.approved_domains = [
                d.strip() for d in domains_input.split(",") if d.strip()
            ]

    if config.approve_interval is not None:
        approve.interval = config.approve_interval
    elif not config.quiet:
        approve.interval = click.prompt(
            "Check interval (seconds)",
            type=int,
            default=approve.interval,
        )


def _print_summary(config_path: Path) -> None:
    """Print setup complete summary."""
    _print_section("SETUP COMPLETE")
    click.echo(f"Config saved: {config_path}")
    click.echo()
    click.echo("To approve scripts for data scientists:")
    click.echo(
        "  syft-bg set-script <script.py> --peers email1@example.com email2@example.com"
    )
    click.echo()
    click.echo("Available commands:")
    click.echo("  syft-bg status     - Show service status")
    click.echo("  syft-bg start      - Start all services")
    click.echo("  syft-bg stop       - Stop all services")
    click.echo("  syft-bg logs <svc> - View service logs")
    click.echo()


def run_init_flow(
    user_passed_config: UserPassedConfig | None = None,
) -> None:
    """Run unified setup for all background services.

    Args:
        user_passed_config: Configuration object with all settings.

    Raises:
        InitFlowError: If the flow cannot proceed.
    """
    if user_passed_config is None:
        user_passed_config = UserPassedConfig()

    if user_passed_config.quiet:
        user_passed_config.skip_oauth = True

    _print_banner(quiet=user_passed_config.quiet)

    config_path = get_default_paths().config

    result_config = _load_existing_config(user_passed_config, config_path)

    _print_section("COMMON SETTINGS", quiet=user_passed_config.quiet)
    _resolve_common_settings(user_passed_config, result_config)

    # _print_section("GMAIL AUTHENTICATION", quiet=user_passed_config.quiet)
    # _setup_auth(user_passed_config, result_config)

    _print_section("NOTIFICATION SERVICE", quiet=user_passed_config.quiet)
    _resolve_notify_settings(user_passed_config, result_config)

    _print_section("AUTO-APPROVAL SERVICE", quiet=user_passed_config.quiet)
    _resolve_approve_settings(user_passed_config, result_config)

    result_config.save(config_path)

    if not user_passed_config.quiet:
        _print_summary(config_path)
