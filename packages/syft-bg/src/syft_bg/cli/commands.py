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
def init(email: str, syftbox_root: str | None, token_path: str | None):
    """Initialize syft-bg configuration.

    Sets up the config file with the data owner email and optional
    SyftBox root directory and OAuth token.

    Examples:

      syft-bg init -e alice@uni.edu

      syft-bg init -e alice@uni.edu -r ~/syftbox -t ~/token.json
    """
    from syft_bg.api.api import init as api_init

    api_init(
        do_email=email,
        syftbox_root=syftbox_root,
        token_path=token_path,
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


def _warn_unmatchable_object(file_contents: list[str]) -> None:
    """Warn about content paths that no submitted job can match."""
    from syft_bg.api.utils import (
        PER_JOB_FILES,
        RUN_SCRIPT_PATH,
        SUBMISSION_ROOT_FILES,
    )

    content = set(file_contents)
    if RUN_SCRIPT_PATH not in content:
        click.echo(
            f"\nWarning: nothing matches the content of '{RUN_SCRIPT_PATH}', the "
            f"file the runner executes, so this object approves no job. Pass it "
            f"in CONTENTS, or build the object from a job you have reviewed with "
            f"syft_bg.auto_approve_job(job).",
            err=True,
        )
    bare = sorted(
        path
        for path in content
        if "/" not in path and path not in SUBMISSION_ROOT_FILES
    )
    if bare:
        click.echo(
            f"\nWarning: {bare} is stored under that name alone, and a job is "
            f"matched on a path relative to its submission root, such as "
            f"'code/main.py'. Pass -b <submission directory> and name the file "
            f"by that path.",
            err=True,
        )
    per_job = sorted(content & PER_JOB_FILES)
    if per_job:
        click.echo(
            f"\nWarning: {per_job} is matched by content, and its bytes carry "
            f"the job name and the time it was submitted, so this object "
            f"approves one job and no other. Pass it with -f instead.",
            err=True,
        )


def _confirm_no_peers() -> None:
    from syft_bg.api.utils import NO_PEERS_WARNING

    click.echo(NO_PEERS_WARNING, err=True)
    click.confirm("Continue?", abort=True)


@main.command("auto-approve")
@click.argument("contents", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--peers",
    "-p",
    multiple=True,
    help="Peer email(s) to restrict to. Can be specified multiple times.",
)
@click.option(
    "--name",
    "-n",
    default=None,
    help="Name for the auto-approval object. Auto-generated if not provided.",
)
@click.option(
    "--file-paths",
    "-f",
    multiple=True,
    help="Filenames to allow by name only (e.g. params.json).",
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
    help="Allow an object without peers (any peer matches) without asking.",
)
def auto_approve(
    contents: tuple[str, ...],
    peers: tuple[str, ...],
    name: str | None,
    file_paths: tuple[str, ...],
    base_dir: str | None,
    allow_any_peer: bool,
):
    """Create or update an auto-approval object.

    Accepts file paths as contents. These are files whose content will be
    hashed and matched. Without -b, a directory is expanded to the files within
    it, each stored relative to that directory, while a single file is stored
    under its base name alone. With -b, name each file by the path a job is
    matched on.

    A job is matched on paths relative to its submission root, so code sits
    under "code/" and the script the runner executes is "run.sh". The rule must
    name every file of the submission and pin the content of run.sh, or it
    matches nothing. To build that from a job you have reviewed, use
    syft_bg.auto_approve_job(job) instead of this command.

    Name config.yaml with -f. Its bytes carry the job name and the time it was
    submitted, so a rule that hashes it matches one job and no other. A
    directory argument, which needs no -b, hashes every file it finds,
    config.yaml included, so it has the same effect.

    A name is read from the current directory as well as from -b, so run this
    from inside the submission.

    Examples, for a submission holding code/main.py, run.sh and config.yaml:

      cd ./job
      syft-bg auto-approve code/main.py run.sh -b . -f config.yaml -p alice@uni.edu

      syft-bg auto-approve code/main.py run.sh -b . -f config.yaml -n my_analysis

    And for one that also holds code/params.json:

      syft-bg auto-approve code/main.py run.sh -b . -f code/params.json -f config.yaml
    """
    from pathlib import Path

    from syft_bg.api.api import auto_approve as api_auto_approve

    if not peers and not allow_any_peer:
        _confirm_no_peers()

    result = api_auto_approve(
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

    click.echo(f"Auto-approval object: {result.name}")
    if result.file_contents:
        for entry in result.file_contents:
            click.echo(f"  {entry}")
    if result.peers:
        click.echo(f"Peers: {', '.join(result.peers)}")
    else:
        click.echo("Peers: (any)")
    if result.file_paths:
        click.echo(f"Allowed files: {', '.join(result.file_paths)}")

    _warn_unmatchable_object(result.file_contents or [])


@main.command("remove-auto-approval")
@click.argument("files", nargs=-1, required=True)
@click.option(
    "--name",
    "-n",
    required=True,
    help="Name of the auto-approval object to remove scripts from.",
)
def remove_auto_approval(files: tuple[str, ...], name: str):
    """Remove scripts from an auto-approval object.

    A stored path is relative to the job submission root, so code sits under
    "code/".

    Examples:

      syft-bg remove-auto-approval code/utils.py -n my_analysis

      syft-bg remove-auto-approval code/main.py code/utils.py -n my_analysis
    """
    from syft_bg.api.utils import get_api_store

    store = get_api_store()
    if not store.exists(name):
        click.echo(f"Auto-approval object '{name}' not found.", err=True)
        raise SystemExit(1)

    obj = store.get(name)
    before = len(obj.file_contents)
    obj.file_contents = [s for s in obj.file_contents if s.relative_path not in files]
    removed = before - len(obj.file_contents)
    store.save(name, obj)

    click.echo(f"Removed {removed} script(s) from '{name}'.")


@main.command("remove-peer")
@click.argument("peer")
@click.option(
    "--name",
    "-n",
    default=None,
    help="Remove peer from a specific object only. If not given, removes from all.",
)
@click.option(
    "--allow-any-peer",
    is_flag=True,
    help="Allow leaving an object without peers (any peer matches) without asking.",
)
def remove_peer(peer: str, name: str | None, allow_any_peer: bool):
    """Remove a peer from auto-approval objects.

    An object left without peers matches any peer and is readable by
    everyone, so you are asked to confirm that unless --allow-any-peer is given.

    Examples:

      syft-bg remove-peer alice@uni.edu

      syft-bg remove-peer alice@uni.edu -n my_analysis
    """
    from syft_bg.api.utils import get_api_store

    store = get_api_store()
    if name and not store.exists(name):
        click.echo(f"Auto-approval object '{name}' not found.", err=True)
        raise SystemExit(1)

    names = [name] if name else store.names()
    objects = {n: store.get(n) for n in names}
    objects = {n: obj for n, obj in objects.items() if peer in obj.peers}
    if not objects:
        click.echo(f"Peer {peer} not found in any auto-approval object.", err=True)
        raise SystemExit(1)

    if not allow_any_peer and any(obj.peers == [peer] for obj in objects.values()):
        _confirm_no_peers()

    for obj_name, obj in objects.items():
        obj.peers.remove(peer)
        store.save(obj_name, obj)

    click.echo(f"Removed peer {peer} from {len(objects)} object(s).")


@main.command("list-auto-approvals")
@click.option(
    "--name",
    "-n",
    default=None,
    help="Show a specific auto-approval object only.",
)
def list_auto_approvals(name: str | None):
    """List auto-approval objects and their scripts.

    Examples:

      syft-bg list-auto-approvals

      syft-bg list-auto-approvals -n my_analysis
    """
    from syft_bg.api.api import list_auto_approvals as api_list_auto_approvals

    objects = api_list_auto_approvals()

    if not objects:
        click.echo("No auto-approval objects configured.")
        return

    if name:
        if name not in objects:
            click.echo(f"Auto-approval object '{name}' not found.", err=True)
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
