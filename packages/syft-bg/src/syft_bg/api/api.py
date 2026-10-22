"""Pythonic API for syft-bg initialization and configuration."""

import shutil
from collections.abc import Sequence
from pathlib import Path

from syft_job.policies import (
    PolicyResolutionError,
    PolicyResult,
    PolicyRunner,
    PolicyState,
)
from syft_rds.apis import ApiArg

from syft_bg.api.results import ApiResult, InstallationResult, StatusResult
from syft_bg.api.utils import (
    confirm_no_peers,
    load_apis_or_empty,
    api_input_error,
    default_api_name,
    default_disclosures,
    infer_params_file,
    api_store_for_job,
    get_api_store,
    new_policy_work_dir,
    policy_kind_for_test,
    policy_test_inputs,
    policy_test_state_dir,
    policy_under_test,
    read_job_args,
    get_job_user_files,
    move_token_to_syftbg_dir,
    resolve_content_files,
    resolve_job_approval_files,
    setup_orchestrator,
)
from syft_bg.approve.api_store import (
    UNCHANGED,
    ApiExistsError,
    ApiNotFoundError,
    ApiStore,
)
from syft_rds.apis.models import ApiDefinition
from syft_bg.common.config import get_default_paths, get_syftbg_dir
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
    token_kwargs: dict = {}
    if token_path is not None:
        # A single OAuth token is used for both the Gmail and Drive APIs.
        token_path = move_token_to_syftbg_dir(Path(token_path))
        token_kwargs = {"drive_token_path": token_path, "gmail_token_path": token_path}

    config = SyftBgConfig(
        do_email=do_email,
        syftbox_root=syftbox_root,
        **token_kwargs,
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
    config_path = get_default_paths().config
    if not config_path.exists():
        print("No config file found, run init first")
        return

    if isinstance(services, list):
        services = {name: {} for name in services}
    manager = ServiceManager()

    # store new settings
    with SyftBgConfig.edit(config_path) as config:
        for name, service_config in services.items():
            service = manager.get_service(name)
            if not service:
                raise ValueError(f"Unknown service: {name}")
            config.set_service_config(name, service_config)

    # make sure services are running
    for name, service_config in services.items():
        service = manager.get_service(name)

        # service was already confirmed to exist above, so ignore type checker here
        if service.is_running() and not restart:  # ty:ignore[unresolved-attribute]
            print(
                f"{name} is already running, skipping. If you want to restart it, set restart=True."
            )
            continue
        elif service.is_running() and restart:  # ty:ignore[unresolved-attribute]
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


def logs(service: str, n: int = 50, as_list: bool = False) -> list[str] | None:
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
        apis=load_apis_or_empty(config),
        service_infos=manager.get_service_infos(),
        is_colab=is_colab(),
    )


# ---------------------------------------------------------------------------
# Apis
# ---------------------------------------------------------------------------


def create_api(
    contents: Sequence[str | Path],
    file_paths: list[str] | None = None,
    peers: list[str] | None = None,
    name: str | None = None,
    base_dir: Path | None = None,
    allow_any_peer: bool = False,
    api_store: ApiStore | None = None,
    args: list[ApiArg | dict] | None = None,
    input_policy: str | Path | None = None,
    output_policy: str | Path | None = None,
    disclosures: list[str] | None = None,
    params_file: str | None = None,
) -> ApiResult:
    """Create an api in SyftBox, which peers call with client.api.<name>(...).

    Copies the files to <datasite>/app_data/apis/<name>/, computes hashes, and
    grants read access to the peers that can call it. A call sends exactly the
    files the api takes by name; the DO's client adds the pinned files and
    approves the call.

    An api pins run.sh (and the rest of its code) by content, and takes
    config.yaml and any other files a call sends by name. No file is both.
    `create_api_from_job` builds that shape from a reviewed job.

    Args:
        contents: List of file paths to pin by content. When base_dir is set,
                  these are relative paths resolved against it. Otherwise,
                  absolute paths or directories (expanded to all files within).
        file_paths: Relative paths a call sends, taken by name only, e.g.
                  ["config.yaml", "code/params.json", "code/data.csv"]. A call
                  must send every one of them, and nothing else.
        peers: Peer emails that may call the api. If None or empty, any peer
               may, and everyone can read the api; you are asked to confirm.
        name: Name for the api. Auto-generated if not provided.
        base_dir: Base directory to resolve relative paths in contents against.
                  When set, FileEntry.relative_path stores the relative path.
        allow_any_peer: Allow no peers without the confirmation prompt.
        api_store: Where to store the api. Defaults to the datasite in config.yaml.
        args: Arguments of the params file, in the order a caller passes them
              positionally: ApiArg(name, type, default, required) or the
              same as a dict.
        input_policy: A .py or .sh script that runs before each call's job.
              A non-zero exit stops the job. See syft_job/policies/env.py
              for what a policy gets and the conventions it follows.
        output_policy: A .py or .sh script that runs after each call's job
              and decides what is shared and with whom. See env.py too.
        disclosures: The items in DisclosureItem that a call's approval
              releases, e.g. ["logs"]. Defaults to default_disclosures in
              config.yaml.
        params_file: The file in file_paths that client.api builds from a
              call's arguments. Defaults to the one JSON file under code/ in
              file_paths, if there is exactly one.

    Returns:
        ApiResult with the created api's details.
    """
    file_paths = file_paths or []
    peers = peers or []
    shape = ApiDefinition(
        file_paths=file_paths,
        args=[ApiArg.model_validate(a) for a in args or []],
        params_file=params_file or infer_params_file(file_paths),
    )
    content_files, error = resolve_content_files(contents, base_dir)
    error = error or api_input_error(
        content_files, shape, [input_policy, output_policy]
    )
    if error:
        return ApiResult(success=False, error=error)
    if not peers and not allow_any_peer and not confirm_no_peers():
        return ApiResult(success=False, error="Aborted: no peers given")
    try:
        name, obj = (api_store or get_api_store()).create(
            default_api_name(name, content_files),
            content_files,
            file_paths,
            peers,
            shape.args,
            input_policy=Path(input_policy) if input_policy else None,
            output_policy=Path(output_policy) if output_policy else None,
            disclosures=default_disclosures() if disclosures is None else disclosures,
            params_file=shape.params_file,
        )
    except (ApiExistsError, ValueError) as e:
        return ApiResult(success=False, error=str(e))
    return ApiResult.from_api(name, obj)


def create_api_from_job(
    job,
    contents: list[str] | None = None,
    file_paths: list[str] | None = None,
    peers: list[str] | None = None,
    name: str | None = None,
    allow_any_peer: bool = False,
    args: list[ApiArg | dict] | None = None,
    input_policy: str | Path | None = None,
    output_policy: str | Path | None = None,
    disclosures: list[str] | None = None,
) -> ApiResult:
    """Create an api from a job you have reviewed.

    Extracts files from the job and routes them to create_api() based on
    the contents and file_paths parameters. Paths are relative to the job
    submission root, so the api pins "run.sh", the file the runner
    executes. A bare name is resolved against the job.

    Args:
        job: JobInfo object to use as template.
        contents: Filenames from the job to match by name AND content.
                  When None, every file is content-matched except config.yaml
                  and code/params.json, which are matched by name.
        file_paths: Filenames from the job to match by name only.
        peers: Peer emails to restrict to. If None, defaults to the job's submitter.
        name: Name for the api. Defaults to the job name.
        allow_any_peer: Allow an empty peers list without the prompt.
        args: Arguments of the api. When None, names, types and defaults are
              inferred from the job's params json.
        input_policy: A .py or .sh script that runs before each call's job.
        output_policy: A .py or .sh script that runs after each call's job.
              See create_api, and syft_job/policies/env.py for what a
              policy gets and the conventions it follows.
        disclosures: What a call's approval releases. See create_api.

    Returns:
        ApiResult with the created object details.
    """
    if peers is None:
        peers = [job.submitted_by]

    content_rel_paths, name_only, error = resolve_job_approval_files(
        get_job_user_files(job), contents, file_paths
    )
    if error:
        return ApiResult(success=False, error=error)

    if name is None:
        name = job.name

    return create_api(
        contents=content_rel_paths,
        file_paths=name_only,
        peers=peers,
        name=name,
        base_dir=job.job_submission_path,
        allow_any_peer=allow_any_peer,
        api_store=api_store_for_job(job),
        args=read_job_args(job, name_only) if args is None else args,
        input_policy=input_policy,
        output_policy=output_policy,
        disclosures=disclosures,
    )


def set_policies(
    name: str,
    input_policy: str | Path | None = UNCHANGED,
    output_policy: str | Path | None = UNCHANGED,
    reset_state: bool = False,
    api_store: ApiStore | None = None,
) -> ApiResult:
    """Add, replace or remove the policies of an existing api.

    Leave a policy out to keep it, pass None to remove it. Replacing a policy
    keeps the api's state; pass reset_state=True to clear it. See
    syft_job/policies/env.py for what a policy gets and the conventions it
    follows.
    """
    for policy in (input_policy, output_policy):
        if isinstance(policy, (str, Path)) and not Path(policy).expanduser().is_file():
            return ApiResult(success=False, error=f"No policy at {policy}")
    try:
        obj = (api_store or get_api_store()).set_policies(
            name,
            input_policy=_as_path(input_policy),
            output_policy=_as_path(output_policy),
            reset_state=reset_state,
        )
    except (ApiNotFoundError, ValueError) as e:
        return ApiResult(success=False, error=str(e))
    return ApiResult.from_api(name, obj)


def run_policy(
    script: str | Path | None = None,
    api: str | None = None,
    kind: str | None = None,
    params: dict | None = None,
    submitter: str = "ds@example.com",
    state_dir: str | Path | None = None,
    outputs_dir: str | Path | None = None,
    candidates_dir: str | Path | None = None,
    api_store: ApiStore | None = None,
) -> PolicyResult:
    """Run a policy on its own, the way the job runner runs it.

    No job, peer or sync is needed. Pass a script on disk, or api=<name> and
    kind ("input" or "output") to run an existing api's policy against a copy
    of its real state, so the real state is not changed.

    Args:
        script: A .py or .sh policy file.
        api: Name of an existing api, instead of a script.
        kind: "input" or "output". Defaults to "output" when sample outputs
              or candidates are given, else "input".
        params: The call's params, as the policy reads them.
        submitter: The data scientist's email the policy sees.
        state_dir: A state folder to use and keep, e.g. to run a rate limit
              several times. Defaults to a throwaway copy.
        outputs_dir: Sample outputs (output policy). A copy is used.
        candidates_dir: Sample output_candidates/ next to outputs/.

    Returns:
        PolicyResult: passed, error (what the data scientist would see), state
        (state.json after the run), recipients and outputs_dir (output policy).

    See syft_job/policies/env.py for what a policy gets and the conventions it
    follows.
    """
    policy_kind = policy_kind_for_test(kind, outputs_dir, candidates_dir)
    work_dir = new_policy_work_dir()
    try:
        policy, api_name, real_state = policy_under_test(
            work_dir, policy_kind, script, api, api_store
        )
    except PolicyResolutionError as e:
        return PolicyResult(passed=False, error=str(e))
    state = PolicyState(policy_test_state_dir(work_dir, state_dir, real_state))
    inputs = policy_test_inputs(
        work_dir, policy_kind, api_name, submitter, params, outputs_dir, candidates_dir
    )
    result = PolicyRunner().run(policy, state, inputs)
    result.outputs_dir = inputs.outputs_dir
    return result


def _as_path(policy: str | Path | None | object) -> Path | None | object:
    return Path(policy) if isinstance(policy, str) else policy


def list_apis() -> dict[str, ApiDefinition]:
    """Return all apis stored in SyftBox.

    Returns:
        Mapping of name → ApiDefinition.
    """
    return get_api_store().load_all()


def remove_api(name: str) -> ApiResult:
    """Remove an api by name.

    Deletes the api folder from SyftBox. A call that arrives later is
    rejected, and an approved call whose api is gone fails before it runs.

    Args:
        name: Name of the api to remove.

    Returns:
        ApiResult with success/error status.
    """
    if not get_api_store().delete(name):
        return ApiResult(success=False, error=f"Api '{name}' not found")
    return ApiResult(success=True, name=name)
