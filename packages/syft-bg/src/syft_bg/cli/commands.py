"""CLI commands for syft-bg."""

import traceback
from typing import Optional

import click


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """SyftBox Background Services Manager."""
    if ctx.invoked_subcommand is None:
        # Default to TUI dashboard if no command given
        ctx.invoke(tui)


@main.command()
def status():
    """Show status of all services."""
    from syft_bg.api.api import status as api_status

    click.echo(api_status().render(as_html=False))


@main.command()
@click.argument("service", required=False)
def start(service: Optional[str]):
    """Start services. If SERVICE specified, start only that service."""
    from syft_bg.api.api import start as api_start

    api_start(service=service)


@main.command()
@click.argument("service", required=False)
def stop(service: Optional[str]):
    """Stop services. If SERVICE specified, stop only that service."""
    from syft_bg.api.api import stop as api_stop

    api_stop(service=service)


@main.command()
@click.argument("service", required=False)
def restart(service: Optional[str]):
    """Restart services. If SERVICE specified, restart only that service."""
    from syft_bg.api.api import restart as api_restart

    api_restart(service=service)


@main.command()
@click.argument("service")
@click.option("--follow", "-f", is_flag=True, help="Follow log output")
@click.option("--lines", "-n", type=int, default=50, help="Number of lines to show")
def logs(service: str, follow: bool, lines: int):
    """View logs for a service."""
    from syft_bg.api.api import logs as api_logs

    api_logs(service=service, follow=follow, lines=lines)


@main.command()
def tui():
    """Launch interactive TUI dashboard."""
    from syft_bg.tui import SyftBgApp

    app = SyftBgApp()
    result = app.run()

    # Handle special exit codes
    if result == 2:
        from syft_bg.cli.init import InitFlowError, run_init_flow

        try:
            run_init_flow()
        except InitFlowError as e:
            click.echo(f"Error: {e}")


@main.command("setup-status")
def setup_status():
    """Check environment and show setup status.

    Verifies that all required credentials and tokens are in place.

    Examples:

      syft-bg setup-status
    """
    from syft_bg.common.config import get_syftbg_dir
    from syft_bg.common.drive import is_colab

    creds_dir = get_syftbg_dir()

    click.echo()
    click.echo("SYFT-BG ENVIRONMENT CHECK")
    click.echo("=" * 50)
    click.echo()

    issues = []

    # Check credentials.json
    credentials_path = creds_dir / "credentials.json"
    click.echo("Checking credentials...")
    if credentials_path.exists():
        click.echo(f"  ✓ credentials.json found at {credentials_path}")
    else:
        click.echo(f"  ✗ credentials.json MISSING at {credentials_path}")
        issues.append(
            "Missing credentials.json:\n"
            "  1. Go to Google Cloud Console → APIs & Services → Credentials\n"
            "  2. Create OAuth 2.0 Client ID (Desktop app)\n"
            "  3. Download as credentials.json\n"
            f"  4. Place at: {credentials_path}"
        )

    # Check authentication tokens
    click.echo()
    click.echo("Checking authentication tokens...")

    gmail_token_path = creds_dir / "gmail_token.json"
    if gmail_token_path.exists():
        click.echo(f"  ✓ Gmail token: {gmail_token_path}")
    else:
        click.echo("  ✗ Gmail token: MISSING")
        issues.append(
            "Missing Gmail token:\n"
            "  Run 'syft-bg init' to complete Gmail authentication"
        )

    if not is_colab():
        drive_token_path = creds_dir / "drive_token.json"
        if drive_token_path.exists():
            click.echo(f"  ✓ Drive token: {drive_token_path}")
        else:
            click.echo("  ✗ Drive token: MISSING")
            issues.append(
                "Missing Drive token:\n"
                "  Run 'syft-bg init' to complete Drive authentication"
            )
    else:
        click.echo("  ✓ Drive token: Colab (native)")

    # Check configuration
    click.echo()
    click.echo("Checking configuration...")

    config_path = creds_dir / "config.yaml"
    if config_path.exists():
        click.echo(f"  ✓ Config file: {config_path}")
        # Try to load and show email
        try:
            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            if "do_email" in config:
                click.echo(f"  ✓ Email: {config['do_email']}")
            if "syftbox_root" in config:
                click.echo(f"  ✓ SyftBox root: {config['syftbox_root']}")
        except Exception:
            pass
    else:
        click.echo("  ✗ Config file: MISSING")
        issues.append(
            "Missing config file:\n  Run 'syft-bg init' to create configuration"
        )

    # Summary
    click.echo()
    click.echo("-" * 50)

    if issues:
        click.echo(f"⚠️  {len(issues)} issue(s) found")
        click.echo()
        for issue in issues:
            click.echo(issue)
            click.echo()
    else:
        click.echo("✅ Environment ready! Run 'syft-bg start' to begin.")

    click.echo()


@main.command("run-foreground")
@click.option(
    "--service",
    "-s",
    type=click.Choice(["notify", "approve", "email_approve", "sync"]),
    required=True,
    help="Service to run",
)
@click.option("--once", is_flag=True, help="Run single check cycle and exit")
def run_foreground(service: str, once: bool):
    """Run a service in foreground.

    This command is used internally by 'syft-bg start' to spawn services
    as subprocesses. You can also use it directly for debugging.

    Examples:

      syft-bg run-foreground --service notify

      syft-bg run-foreground --service approve --once
    """
    from syft_bg.api.api import run_foreground as api_run_foreground

    try:
        api_run_foreground(service=service, once=once)
    except FileNotFoundError:
        click.echo(
            f"Error initializing service {service}: {traceback.format_exc()}", err=True
        )
        click.echo("Run 'syft-bg init' first to configure the service.", err=True)
        raise SystemExit(1)


@main.command()
@click.option(
    "--email",
    "-e",
    required=True,
    help="Data owner email address.",
)
@click.option(
    "--syftbox-root",
    "-r",
    default=None,
    type=click.Path(),
    help="Path to the SyftBox root directory.",
)
@click.option(
    "--token-path",
    "-t",
    default=None,
    type=click.Path(exists=True),
    help="Path to the OAuth token file.",
)
@click.option(
    "--encryption",
    type=click.Choice(["off", "optional", "required"]),
    default="optional",
    show_default=True,
    help="Encryption policy. Use the same policy as the notebook login.",
)
def init(email: str, syftbox_root: str | None, token_path: str | None, encryption: str):
    """Initialize syft-bg configuration.

    Sets up the config file with the data owner email and optional
    SyftBox root directory and OAuth token.

    Examples:

      syft-bg init -e alice@uni.edu

      syft-bg init -e alice@uni.edu -r ~/syftbox -t ~/token.json

      syft-bg init -e alice@uni.edu --encryption required
    """
    from syft_bg.api.api import init as api_init

    api_init(
        do_email=email,
        syftbox_root=syftbox_root,
        token_path=token_path,
        encryption=encryption,
    )


@main.command("ensure-running")
@click.argument("services", nargs=-1, required=True)
@click.option(
    "--restart", is_flag=True, help="Restart services even if already running"
)
@click.option(
    "--install", is_flag=True, help="Install systemd service units for autostart"
)
def ensure_running(services: tuple[str, ...], restart: bool, install: bool):
    """Start services if they aren't already running.

    Examples:

      syft-bg ensure-running notify approve

      syft-bg ensure-running notify --restart

      syft-bg ensure-running notify approve --install
    """
    from syft_bg.api.api import ensure_running as api_ensure_running

    api_ensure_running(list(services), restart=restart, install=install)


def _confirm_no_peers() -> None:
    from syft_bg.api.utils import NO_PEERS_WARNING

    click.echo(NO_PEERS_WARNING, err=True)
    click.confirm("Continue?", abort=True)


@main.command("create-api")
@click.argument("contents", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--peers",
    "-p",
    multiple=True,
    help="Peer email(s) that may call the api. Can be specified multiple times.",
)
@click.option(
    "--name",
    "-n",
    default=None,
    help="Name for the api. Auto-generated if not provided.",
)
@click.option(
    "--file-paths",
    "-f",
    multiple=True,
    help="Files a call sends, taken by name only (e.g. code/params.json).",
)
@click.option(
    "--base-dir",
    "-b",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Base directory to resolve relative paths in contents against.",
)
@click.option(
    "--allow-any-peer",
    is_flag=True,
    help="Allow an api without peers (any peer may call it) without asking.",
)
def create_api(
    contents: tuple[str, ...],
    peers: tuple[str, ...],
    name: str | None,
    file_paths: tuple[str, ...],
    base_dir: str | None,
    allow_any_peer: bool,
):
    """Create an api that peers call with client.api.<name>(...).

    CONTENTS are the files pinned by content: run.sh and the rest of the code.
    -f names the files a call sends, such as the JSON file under code/ that
    holds the arguments. No file is in both; config.yaml is in neither, since
    the data owner writes it for each call. Paths
    are relative to a job's submission root, so pass -b <submission directory>
    and name each file by that path.
    To build an api from a job you have reviewed, use
    syft_bg.create_api_from_job(job) instead of this command.

    Example, for a submission holding code/main.py, code/params.json, run.sh
    and config.yaml:

      cd ./job
      syft-bg create-api code/main.py run.sh -b . -f code/params.json -p alice@uni.edu
    """
    from pathlib import Path

    from syft_bg.api.api import create_api as api_create_api

    if not peers and not allow_any_peer:
        _confirm_no_peers()

    result = api_create_api(
        contents=list(contents),
        file_paths=list(file_paths) or None,
        peers=list(peers) or None,
        name=name,
        base_dir=Path(base_dir) if base_dir else None,
        allow_any_peer=True,
    )
    if not result.success:
        click.echo(f"Error: {result.error}", err=True)
        raise SystemExit(1)
    click.echo(repr(result))


@main.command("remove-api")
@click.argument("name")
def remove_api(name: str):
    """Remove an api.

    Example:

      syft-bg remove-api my_analysis
    """
    from syft_bg.api.api import remove_api as api_remove_api

    result = api_remove_api(name)
    if not result.success:
        click.echo(f"Error: {result.error}", err=True)
        raise SystemExit(1)
    click.echo(f"Removed api '{name}'.")


@main.command("remove-peer")
@click.argument("peer")
@click.option(
    "--name",
    "-n",
    default=None,
    help="Remove peer from a specific api only. If not given, removes from all.",
)
@click.option(
    "--allow-any-peer",
    is_flag=True,
    help="Allow leaving an api without peers (any peer may call it) without asking.",
)
def remove_peer(peer: str, name: str | None, allow_any_peer: bool):
    """Remove a peer from apis.

    An api left without peers may be called by any peer and is readable by
    everyone, so you are asked to confirm that unless --allow-any-peer is given.

    Examples:

      syft-bg remove-peer alice@uni.edu

      syft-bg remove-peer alice@uni.edu -n my_analysis
    """
    from syft_bg.api.utils import get_api_store

    store = get_api_store()
    if name and not store.exists(name):
        click.echo(f"Api '{name}' not found.", err=True)
        raise SystemExit(1)

    names = [name] if name else store.names()
    objects = {n: store.get(n) for n in names}
    objects = {n: obj for n, obj in objects.items() if peer in obj.peers}
    if not objects:
        click.echo(f"Peer {peer} not found in any api.", err=True)
        raise SystemExit(1)

    if not allow_any_peer and any(obj.peers == [peer] for obj in objects.values()):
        _confirm_no_peers()

    for obj_name, obj in objects.items():
        obj.peers.remove(peer)
        store.save(obj_name, obj)

    click.echo(f"Removed peer {peer} from {len(objects)} api(s).")


@main.command("list-apis")
@click.option(
    "--name",
    "-n",
    default=None,
    help="Show a specific api only.",
)
def list_apis(name: str | None):
    """List apis and their files.

    Examples:

      syft-bg list-apis

      syft-bg list-apis -n my_analysis
    """
    from syft_bg.api.api import list_apis as api_list_apis

    objects = api_list_apis()

    if not objects:
        click.echo("No apis created.")
        return

    if name:
        if name not in objects:
            click.echo(f"Api '{name}' not found.", err=True)
            raise SystemExit(1)
        objects_to_show = {name: objects[name]}
    else:
        objects_to_show = objects

    for obj_name, obj in objects_to_show.items():
        click.echo(f"\n[{obj_name}]")
        if obj.file_contents:
            click.echo("  File contents:")
            for entry in obj.file_contents:
                click.echo(f"    {entry.relative_path:<30} {entry.hash}")
        else:
            click.echo("  File contents: (none)")
        if obj.file_paths:
            click.echo(f"  Allowed files: {', '.join(obj.file_paths)}")
        if obj.peers:
            click.echo(f"  Peers: {', '.join(obj.peers)}")
        else:
            click.echo("  Peers: (any)")
    click.echo()


@main.command()
@click.argument("service", required=False)
def install(service: Optional[str]):
    """Install syft-bg systemd user service(s).

    If SERVICE is given, install only that service. Otherwise install all.

    Examples:

      syft-bg install

      syft-bg install notify
    """
    from syft_bg.api.api import install as api_install

    label = service or "all services"
    click.echo(f"Installing {label}...")
    results = api_install(service)

    failed = False
    for r in results:
        if r.success:
            click.echo(f"  ✅ {r.service}: {r.message}")
        else:
            click.echo(f"  ❌ {r.service}: {r.message}", err=True)
            failed = True

    if failed:
        raise SystemExit(1)


@main.command()
@click.argument("service", required=False)
def uninstall(service: Optional[str]):
    """Uninstall syft-bg systemd user service(s).

    If SERVICE is given, uninstall only that service. Otherwise uninstall all.

    Examples:

      syft-bg uninstall

      syft-bg uninstall notify
    """
    from syft_bg.api.api import uninstall as api_uninstall

    label = service or "all services"
    click.echo(f"Uninstalling {label}...")
    results = api_uninstall(service)

    failed = False
    for r in results:
        if r.success:
            click.echo(f"  ✅ {r.service}: {r.message}")
        else:
            click.echo(f"  ❌ {r.service}: {r.message}", err=True)
            failed = True

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
