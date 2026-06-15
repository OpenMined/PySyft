"""Pythonic API for syft-bg initialization and configuration."""

import shutil
from pathlib import Path

from syft_bg.api.results import AutoApproveResult, InstallationResult, StatusResult
from syft_bg.api.utils import (
    copy_and_hash_files,
    generate_unique_name,
    get_job_user_files,
    move_token_to_syftbg_dir,
    resolve_auto_approve_file_args,
    resolve_content_files,
    setup_orchestrator,
    validate_auto_approve_job_inputs,
)
from syft_bg.approve.config import AutoApproveConfig, AutoApprovalObj
from syft_bg.common.config import get_syftbg_dir, get_default_paths
from syft_bg.common.drive import is_colab
from syft_bg.common.syft_bg_config import SyftBgConfig
from syft_bg.services import ServiceManager
from syft_bg.systemd import install_service, is_installed, uninstall_service

# This file should only contain user facing methods,
# util functions should be in the utils.py file.


def init(
    do_email: str,
    syftbox_root: str | Path | None = None,
    token_path: str | Path | None = None,
    settings: dict[str, dict] | None = None,
) -> None:
    if token_path is not None:
        token_path = Path(token_path)
        move_token_to_syftbg_dir(token_path)

    config = SyftBgConfig(
        do_email=do_email,
        syftbox_root=syftbox_root,
        token_path=token_path,
        drive_token_path=token_path,
    )

    if settings:
        for service_name, service_settings in settings.items():
            config.set_service_config(service_name, service_settings)

    config.save()
    print(f"Config saved to {get_default_paths().config}")


def ensure_running(
    services: dict[str, dict] | list[str],
    restart: bool = False,
    install: bool = False,
) -> None:
    # store new settings
    try:
        config = SyftBgConfig.from_path()
    except FileNotFoundError:
        print("No config file found, run init first")
        return

    if isinstance(services, list):
        services = {name: {} for name in services}
    manager = ServiceManager()

    # store new settings
    for name, service_config in services.items():
        service = manager.get_service(name)
        if not service:
            raise ValueError(f"Unknown service: {name}")
        config.set_service_config(name, service_config)

    config.save()

    # make sure services are running
    for name, service_config in services.items():
        service = manager.get_service(name)
        if service.is_running() and not restart:
            print(
                f"{name} is already running, skipping. If you want to restart it, set restart=True."
            )
            continue
        elif service.is_running() and restart:
            manager.restart_service(name)
        else:
            manager.start_service(name)
            print(f"{name} started")

    if install:
        for name in services:
            ok, msg = install_service(name)
            if ok:
                print(f"{name} installed: {msg}")
            else:
                print(f"{name} install failed: {msg}")


# ---------------------------------------------------------------------------
# Service control
# ---------------------------------------------------------------------------


def start(service: str | None = None) -> dict[str, tuple[bool, str]]:
    """Start background services.

    Args:
        service: Name of a specific service to start (e.g. "notify", "approve").
                 If None, starts all services.

    Returns:
        Dict mapping service name to (success, message).
    """
    manager = ServiceManager()
    if service:
        return {service: manager.start_service(service)}
    return manager.start_all()


def stop(service: str | None = None) -> dict[str, tuple[bool, str]]:
    """Stop background services.

    Args:
        service: Name of a specific service to stop.
                 If None, stops all services.

    Returns:
        Dict mapping service name to (success, message).
    """
    manager = ServiceManager()
    if service:
        return {service: manager.stop_service(service)}
    return manager.stop_all()


def restart(service: str | None = None) -> dict[str, tuple[bool, str]]:
    """Restart background services.

    Args:
        service: Name of a specific service to restart.
                 If None, restarts all services.

    Returns:
        Dict mapping service name to (success, message).
    """
    manager = ServiceManager()
    if service:
        return {service: manager.restart_service(service)}
    results = {}
    for name in manager.list_services():
        results[name] = manager.restart_service(name)
    return results


def reset() -> None:
    """Stop all services, uninstall systemd units, and clear all state."""
    manager = ServiceManager()

    for name in manager.list_services():
        if is_installed(name):
            ok, msg = uninstall_service(name)
            if not ok:
                print(f"Warning: failed to uninstall {name}: {msg}")

    manager.stop_all()

    syftbg_dir = get_syftbg_dir()
    shutil.rmtree(syftbg_dir, ignore_errors=True)
    print(
        "Reset complete: uninstalled systemd units, stopped services, "
        "cleared state, config, logs, and PID files."
    )


# ---------------------------------------------------------------------------
# Systemd install
# ---------------------------------------------------------------------------


def install(service: str | None = None) -> list[InstallationResult]:
    """Install syft-bg service(s) as systemd user units.

    Args:
        service: Name of a specific service to install.
                 If None, installs all registered services.

    Returns:
        List of InstallationResult, one per service.
    """
    manager = ServiceManager()
    names = [service] if service else manager.list_services()

    results = []
    for name in names:
        if name not in manager.list_services():
            results.append(
                InstallationResult(
                    success=False, service=name, message=f"Unknown service: {name}"
                )
            )
            continue
        ok, msg = install_service(name)
        results.append(InstallationResult(success=ok, service=name, message=msg))
    return results


def uninstall(service: str | None = None) -> list[InstallationResult]:
    """Uninstall syft-bg service(s) systemd user units.

    Args:
        service: Name of a specific service to uninstall.
                 If None, uninstalls all registered services.

    Returns:
        List of InstallationResult, one per service.
    """
    manager = ServiceManager()
    names = [service] if service else manager.list_services()

    results = []
    for name in names:
        if name not in manager.list_services():
            results.append(
                InstallationResult(
                    success=False, service=name, message=f"Unknown service: {name}"
                )
            )
            continue
        ok, msg = uninstall_service(name)
        results.append(InstallationResult(success=ok, service=name, message=msg))
    return results


def logs(service: str, n: int = 50, as_list: bool = False) -> list[str]:
    """Get recent log lines for a service.

    Args:
        service: Service name ("notify" or "approve").
        n: Number of lines to return.

    Returns:
        List of log lines.
    """
    manager = ServiceManager()
    results = manager.get_logs(service, n)
    if as_list:
        return results
    print("\n".join(results))


def run_foreground(service: str, once: bool = False) -> None:
    """Run a service in the foreground (blocking).

    Args:
        service: Service name ("notify", "approve", or "email_approve").
        once: If True, run a single check cycle and exit.
    """
    orchestrator = setup_orchestrator(service)

    if once:
        orchestrator.run_once()
    else:
        orchestrator.run_loop()


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def status() -> StatusResult:
    """Get the current status of syft-bg services and configuration.

    Returns:
        StatusResult with service states, user info, and approval config.

    Example:
        >>> import syft_bg
        >>> syft_bg.status
        syft-bg status
        ========================================
          email:       user@example.com
          syftbox:     ~/SyftBox
          environment: local
          gmail:       ready
        ...
    """
    try:
        config = SyftBgConfig.from_path()
    except FileNotFoundError:
        config = SyftBgConfig()

    manager = ServiceManager()

    return StatusResult(
        config=config,
        service_infos=manager.get_service_infos(),
        is_colab=is_colab(),
    )


# ---------------------------------------------------------------------------
# Auto-approve
# ---------------------------------------------------------------------------


def auto_approve(
    contents: list[str | Path],
    file_paths: list[str] | None = None,
    peers: list[str] | None = None,
    name: str | None = None,
    base_dir: Path | None = None,
) -> AutoApproveResult:
    """Create an auto-approval object.

    Copies files to a managed directory, computes hashes, and saves
    the approval configuration. The approve service will use this to
    automatically approve matching jobs.

    Args:
        contents: List of file paths to approve by content. When base_dir is set,
                  these are relative paths resolved against it. Otherwise,
                  absolute paths or directories (expanded to all files within).
        file_paths: Relative paths to allow by name only (e.g. ["params.json"]).
        peers: Peer emails to restrict to. If None or empty, any peer matches.
        name: Name for the auto-approval object. Auto-generated if not provided.
        base_dir: Base directory to resolve relative paths in contents against.
                  When set, FileEntry.relative_path stores the relative path.

    Returns:
        AutoApproveResult with the created object details.
    """
    if file_paths is None:
        file_paths = []
    if peers is None:
        peers = []

    content_files, error = resolve_content_files(contents, base_dir)
    if error:
        return AutoApproveResult(success=False, error=error)

    if not content_files and not file_paths:
        return AutoApproveResult(success=False, error="No files to process")

    config = AutoApproveConfig.load()
    name = generate_unique_name(name, content_files, config)

    file_entries = copy_and_hash_files(content_files, name)

    obj = AutoApprovalObj(
        file_contents=file_entries,
        file_paths=file_paths,
        peers=peers,
    )
    config.auto_approvals.objects[name] = obj
    config.save()

    return AutoApproveResult(
        success=True,
        name=name,
        file_contents=[e.relative_path for e in file_entries],
        file_paths=file_paths,
        peers=peers,
    )


def auto_approve_job(
    job,
    contents: list[str] | None = None,
    file_paths: list[str] | None = None,
    peers: list[str] | None = None,
    name: str | None = None,
) -> AutoApproveResult:
    """Create an auto-approval config from an existing job.

    Extracts files from the job and routes them to auto_approve() based on
    the contents and file_paths parameters.

    Args:
        job: JobInfo object to use as template.
        contents: Filenames from the job to match by name AND content.
                  If None and file_paths is None, all files are content-matched.
                  If None and file_paths is set, all other files are content-matched.
        file_paths: Filenames from the job to match by name only.
        peers: Peer emails to restrict to. If None, defaults to the job's submitter.
        name: Name for the auto-approval object. Defaults to job name.

    Returns:
        AutoApproveResult with the created object details.
    """
    if peers is None:
        peers = [job.submitted_by]

    user_files = get_job_user_files(job)

    error = validate_auto_approve_job_inputs(user_files, contents, file_paths)
    if error:
        return AutoApproveResult(success=False, error=error)

    content_rel_paths, name_only = resolve_auto_approve_file_args(
        user_files, contents, file_paths
    )

    if name is None:
        name = job.name

    return auto_approve(
        contents=content_rel_paths,
        file_paths=name_only,
        peers=peers,
        name=name,
        base_dir=job.code_dir,
    )


def list_auto_approvals() -> dict[str, AutoApprovalObj]:
    """Return all configured auto-approval objects.

    Returns:
        Mapping of name → AutoApprovalObj.
    """
    return AutoApproveConfig.load().auto_approvals.objects


def remove_auto_approve(name: str) -> AutoApproveResult:
    """Remove an auto-approval object by name.

    Deletes the object from config and removes its copied files from the
    auto-approvals directory. The running approve service will pick up
    the change on its next poll iteration (no restart required).

    Args:
        name: Name of the auto-approval object to remove.

    Returns:
        AutoApproveResult with success/error status.
    """
    config = AutoApproveConfig.load()
    if name not in config.auto_approvals.objects:
        return AutoApproveResult(
            success=False, error=f"Auto-approval object '{name}' not found"
        )

    del config.auto_approvals.objects[name]
    config.save()

    obj_dir = get_default_paths().auto_approvals_dir / name
    if obj_dir.exists():
        shutil.rmtree(obj_dir, ignore_errors=True)

    return AutoApproveResult(success=True, name=name)
