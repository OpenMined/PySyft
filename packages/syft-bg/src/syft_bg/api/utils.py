"""Utility functions used by the syft-bg API layer."""

import os
import shutil
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from syft_bg.approve.api_store import ApiStore
from syft_bg.approve.config import AutoApprovalObj
from syft_bg.approve.criteria import RUN_SCRIPT_PATH, get_job_submission_files
from syft_bg.common.config import get_default_paths, get_syftbg_dir
from syft_bg.common.drive import is_colab
from syft_bg.common.syft_bg_config import SyftBgConfig
from syft_bg.email_approve.pubsub_setup import get_project_id_from_credentials
from syft_bg.services.base import ServiceInfo, ServiceStatus

# config.yaml carries per-job metadata (the job name, the submission time), so
# its bytes are unique to one job. Pinning them makes an object that matches
# that job and no other, so it is never content-matched. It is not executed:
# the runner reads run.sh, which is pinned by content.
PER_JOB_FILES = frozenset({"config.yaml"})

# Matched by name unless the caller says otherwise. params.json is a job's
# parameters, which usually vary run to run.
DEFAULT_NAME_ONLY_FILES = PER_JOB_FILES | {"code/params.json"}

SUBMISSION_ROOT_FILES = frozenset({RUN_SCRIPT_PATH, "config.yaml"})

_GENERATED_DIRS = {".venv", "outputs", "__pycache__"}
NOT_RUN_STATUSES = frozenset({"received", "pending"})


def get_setup_state_path(service: str) -> Path:
    """Get the setup_state.json path for a service."""
    paths = get_default_paths()
    mapping = {
        "notify": paths.notify_setup_state,
        "approve": paths.approve_setup_state,
        "email_approve": paths.email_approve_setup_state,
        "sync": paths.sync_setup_state,
    }
    return mapping[service]


def clear_setup_state(path: Path) -> None:
    """Remove old setup state so stale errors don't persist."""
    if path.exists():
        path.unlink()


def write_setup_state(
    service: str, path: Path, status: ServiceStatus, error: str | None = None
) -> None:
    """Persist a ServiceInfo snapshot to disk."""
    ServiceInfo(name=service, status=status, error=error).save(path)


def setup_orchestrator(service: str):
    """Create an orchestrator, run setup(), and persist state.

    Loads config, builds the orchestrator via from_config, calls setup(),
    and writes the result to setup_state.json. On failure the full
    traceback is captured in the state file.
    """
    import traceback

    from syft_bg.approve import ApprovalOrchestrator
    from syft_bg.common.syft_bg_config import SyftBgConfig
    from syft_bg.email_approve import EmailApproveOrchestrator
    from syft_bg.notify import NotificationOrchestrator
    from syft_bg.sync.orchestrator import SyncOrchestrator

    config = SyftBgConfig.from_path()
    state_path = get_setup_state_path(service)
    clear_setup_state(state_path)
    write_setup_state(service, state_path, ServiceStatus.STARTING)

    try:
        if service == "notify":
            orchestrator = NotificationOrchestrator.from_config(config.notify)
        elif service == "approve":
            orchestrator = ApprovalOrchestrator.from_config(config.approve)
        elif service == "email_approve":
            orchestrator = EmailApproveOrchestrator.from_config(config.email_approve)
        elif service == "sync":
            orchestrator = SyncOrchestrator.from_config(config.sync)
        else:
            raise ValueError(f"Unknown service: {service}")
        orchestrator.setup()
    except Exception:
        write_setup_state(
            service, state_path, ServiceStatus.ERROR, traceback.format_exc()
        )
        raise

    write_setup_state(service, state_path, ServiceStatus.RUNNING)
    return orchestrator


def load_setup_state(service: str) -> ServiceInfo | None:
    """Load persisted ServiceInfo for a service, or None if not found."""
    path = get_setup_state_path(service)
    return ServiceInfo.load(path)


def move_token_to_syftbg_dir(token_path: Path) -> Path:
    syft_bg_dir = get_syftbg_dir()
    syft_bg_dir = Path(syft_bg_dir).expanduser().resolve()
    syft_bg_dir.mkdir(parents=True, exist_ok=True)

    target_token_path = syft_bg_dir / "token.json"
    if Path(token_path).resolve() != target_token_path.resolve():
        if Path(token_path).exists():
            shutil.copy2(token_path, target_token_path)
            token_path = target_token_path
            print(f"Stored token at {target_token_path}")
        else:
            print(f"Warning: Provided token_path ({token_path}) does not exist.")

    return Path(token_path)


def credentials_setup_steps(creds_path: Path, colab: bool) -> str:
    """Return step-by-step instructions for setting up credentials.json."""
    console_url = "https://console.cloud.google.com/apis/credentials"
    if colab:
        save_step = (
            f"  5. Upload the downloaded JSON file to Google Drive at: {creds_path}"
        )
    else:
        save_step = f"  5. Save the downloaded JSON file to: {creds_path}"

    return (
        f"  1. Open Google Cloud Console: {console_url}\n"
        "  2. Create a project (or select an existing one)\n"
        "  3. Click 'Create Credentials' > 'OAuth client ID'\n"
        "     - If prompted, configure the consent screen first\n"
        "       (External type, add your email as a test user)\n"
        "  4. Select 'Desktop app' as application type, then click 'Create'\n"
        f"{save_step}"
    )


def check_credentials_exist(
    credentials_path: Path | None = None,
    gmail_token_path: Path | None = None,
    drive_token_path: Path | None = None,
) -> list[str]:
    """Check that all required credentials and tokens are in place.

    Returns a list of issues. Empty list means all prerequisites are met.
    """
    creds_dir = get_syftbg_dir()
    issues = []
    colab = is_colab()

    # Check credentials.json
    creds_path = (
        Path(credentials_path) if credentials_path else creds_dir / "credentials.json"
    )
    if not creds_path.exists():
        steps = credentials_setup_steps(creds_path, colab)
        issues.append(f"credentials.json not found at {creds_path}\n{steps}")

    # Check Gmail token
    gmail_path = (
        Path(gmail_token_path) if gmail_token_path else creds_dir / "gmail_token.json"
    )
    if not gmail_path.exists():
        if creds_path.exists():
            issues.append(
                f"Gmail token not found at {gmail_path}\n"
                "  Run syft_bg.authenticate() to set it up interactively"
            )
        else:
            issues.append(
                f"Gmail token not found at {gmail_path}\n"
                "  Set up credentials.json first, then run syft_bg.authenticate()"
            )

    # Check Drive token (not needed on Colab — uses native auth)
    if not colab:
        drive_path = (
            Path(drive_token_path)
            if drive_token_path
            else creds_dir / "drive_token.json"
        )
        if not drive_path.exists():
            if creds_path.exists():
                issues.append(
                    f"Drive token not found at {drive_path}\n"
                    "  Run syft_bg.authenticate() to set it up interactively"
                )
            else:
                issues.append(
                    f"Drive token not found at {drive_path}\n"
                    "  Set up credentials.json first, then run syft_bg.authenticate()"
                )

    return issues


def save_gcp_project_id(credentials_path: Path) -> None:
    """Extract project_id from credentials.json and save to config.yaml."""
    try:
        project_id = get_project_id_from_credentials(credentials_path)
        if not get_default_paths().config.exists():
            return
        with SyftBgConfig.edit() as config:
            config.email_approve.gcp_project_id = project_id
    except Exception:
        pass


def default_api_name(name: str | None, content_files: list[tuple[str, Path]]) -> str:
    """Name for a new auto-approval object; ApiStore makes it unique."""
    if name is not None:
        return name
    if len(content_files) == 1:
        return Path(content_files[0][0]).stem
    return "auto_approval"


NO_PEERS_WARNING = (
    "Warning: no peers given. Any peer can run jobs matching this "
    "auto-approval, and everyone can read its files."
)


def confirm_no_peers() -> bool:
    """Warn that an api without peers is public and ask to continue."""
    print(NO_PEERS_WARNING)
    try:
        answer = input("Continue? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


def load_auto_approvals_or_empty(
    config: SyftBgConfig,
) -> dict[str, AutoApprovalObj]:
    """Auto-approvals for status output; empty when syft-bg isn't set up."""
    if not config.do_email or not config.syftbox_root:
        return {}
    return ApiStore(config.syftbox_root, config.do_email).load_all()


def api_store_for_job(job) -> ApiStore:
    """ApiStore for the datasite the job was submitted to."""
    return ApiStore(job._client.config.syftbox_folder, job.datasite_owner_email)


def get_api_store() -> ApiStore:
    """ApiStore for the DO's datasite configured in config.yaml."""
    config = SyftBgConfig.load().approve
    if not config.do_email or not config.syftbox_root:
        raise ValueError(
            "syft-bg is not initialized: config needs do_email and syftbox_root"
        )
    return ApiStore(config.syftbox_root, config.do_email)


def resolve_content_files(
    contents: Sequence[str | Path], base_dir: Path | None
) -> tuple[list[tuple[str, Path]], str | None]:
    """Resolve content paths to (relative_path, absolute_path) pairs.

    Returns (content_files, error). error is None on success.
    """
    content_files: list[tuple[str, Path]] = []
    for item in contents:
        if base_dir is not None:
            # Normalize lexically, so "./code/main.py" is stored as the
            # "code/main.py" a job is matched on.
            abs_path = Path(os.path.abspath(base_dir / item))
            try:
                rel = abs_path.relative_to(os.path.abspath(base_dir)).as_posix()
            except ValueError:
                return [], f"{item} is outside the base directory {base_dir}"
            if not abs_path.exists():
                return [], f"File not found: {abs_path}"
            if abs_path.is_dir():
                return [], (
                    f"{abs_path} is a directory. With a base directory, name "
                    f"each file, so that the stored path is the one a job is "
                    f"matched on."
                )
            content_files.append((rel, abs_path))
        else:
            p = Path(item).expanduser()
            if p.is_dir():
                found = sorted(f for f in p.rglob("*") if f.is_file())
                if not found:
                    return [], f"No files found in {p}"
                for f in found:
                    content_files.append((str(f.relative_to(p)), f))
            elif not p.exists():
                return [], f"File not found: {p}"
            else:
                content_files.append((p.name, p))
    return content_files, None


def resolve_auto_approve_file_args(
    user_files: dict[str, Path],
    contents: list[str] | None,
    file_paths: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Determine which job files are content-matched vs name-only.

    A matcher compares the whole file set, so a job file in neither bucket
    makes the object match nothing. A branch that places the files itself
    therefore places all of them; a branch driven by the caller does not.

    Returns (content_rel_paths, name_only).
    """
    all_files = set(user_files.keys())

    if contents is not None:
        # The caller placed the files itself. auto_approve_job reports one that
        # neither bucket names, rather than guessing where it belongs.
        return list(contents), list(file_paths or [])

    name_only = all_files & DEFAULT_NAME_ONLY_FILES
    if file_paths is not None:
        name_only |= set(file_paths)
    return list(all_files - name_only), list(name_only)


def resolve_job_file_args(
    user_files: dict[str, Path], names: list[str] | None
) -> tuple[list[str] | None, str | None]:
    """Read owner-typed file names as paths relative to the submission root.

    A name that is already a path is kept. Any other name is resolved as a
    path suffix, so "main.py" finds "code/main.py" and "utils/helpers.py" finds
    "code/utils/helpers.py". A name that matches more than one file is an
    error: the owner must say which one.
    """
    if names is None:
        return None, None

    resolved: list[str] = []
    for name in names:
        path_name = PurePosixPath(name)
        if path_name.as_posix() in user_files:
            resolved.append(path_name.as_posix())
            continue
        parts = path_name.parts
        matches = [
            path
            for path in user_files
            if parts and PurePosixPath(path).parts[-len(parts) :] == parts
        ]
        if len(matches) == 1:
            resolved.append(matches[0])
        elif not matches:
            return [], f"File '{name}' not found in job"
        else:
            return [], (
                f"File '{name}' matches several files in the job: "
                f"{sorted(matches)}. Use the full path."
            )
    return resolved, None


def validate_auto_approve_job_inputs(
    user_files: dict[str, Path],
    contents: list[str] | None,
    file_paths: list[str] | None,
) -> str | None:
    """Validate inputs for auto_approve_job. Returns error string or None."""
    if not user_files:
        return "No user files found in job"
    if contents is not None:
        for fname in contents:
            if fname not in user_files:
                return f"File '{fname}' not found in job"
    if file_paths is not None:
        for fname in file_paths:
            if fname not in user_files:
                return f"File '{fname}' not found in job"
    if contents is not None and file_paths is not None:
        overlap = set(contents) & set(file_paths)
        if overlap:
            return f"Overlap between contents and file_paths: {overlap}"
    return None


def validate_auto_approve_object_covers_job(
    user_files: dict[str, Path], content_rel_paths: list[str], name_only: list[str]
) -> str | None:
    """Check that the object being built can approve a job like this one.

    The matcher compares the whole submission tree and demands pinned content
    for run.sh, so an object that misses either one approves nothing, for ever,
    and says so only once a job arrives. Refuse to write it.

    `user_files` leaves out the directories a run creates, and the matcher does
    not. That difference is deliberate: a rule is often built from a job that
    already ran, which carries code/.venv, while the jobs it must approve are
    still pending and carry none. Teaching the matcher the same skip list would
    hide a payload under code/outputs/ from review. Returns an error or None.
    """
    if RUN_SCRIPT_PATH not in user_files:
        return (
            f"the job carries no '{RUN_SCRIPT_PATH}', so it is not a submission "
            f"this version can approve. Check the job directory."
        )

    if RUN_SCRIPT_PATH not in content_rel_paths:
        return (
            f"'{RUN_SCRIPT_PATH}' must be matched by content: it is the file the "
            f"runner executes. Name it in `contents` and not in `file_paths`, or "
            f"leave both unset, which pins every file but "
            f"{sorted(DEFAULT_NAME_ONLY_FILES)}."
        )

    per_job = sorted(set(content_rel_paths) & PER_JOB_FILES)
    if per_job:
        return (
            f"{per_job} cannot be matched by content: the bytes carry the job "
            f"name and the time it was submitted, so no second job would match. "
            f"Name them in `file_paths` instead."
        )

    uncovered = set(user_files) - set(content_rel_paths) - set(name_only)
    if uncovered:
        return (
            f"these files of the job are in neither bucket, so no job would "
            f"match: {sorted(uncovered)}. Name them in `contents` to pin their "
            f"bytes, or in `file_paths` to match them by name."
        )
    return None


def validate_content_files_are_text(
    user_files: dict[str, Path], content_rel_paths: list[str]
) -> str | None:
    """Check that each content-matched file is UTF-8 text.

    The matcher compares content as text and refuses a file it cannot read,
    so a rule that pins such a file approves nothing. Returns an error or None.
    """
    unreadable = []
    for rel in sorted(content_rel_paths):
        try:
            user_files[rel].read_text(encoding="utf-8")
        except UnicodeDecodeError:
            unreadable.append(rel)
    if unreadable:
        return (
            f"these files of the job are not UTF-8 text, so their content "
            f"cannot be matched and no job would match: {unreadable}. A job "
            f"that ships them cannot be auto-approved."
        )
    return None


def resolve_job_approval_files(
    user_files: dict[str, Path],
    contents: list[str] | None,
    file_paths: list[str] | None,
) -> tuple[list[str], list[str], str | None]:
    """Sort the files of a job into content-matched and name-only paths.

    Returns (content_rel_paths, name_only, error). On error, both lists are
    empty and the object must not be written.
    """
    if not user_files:
        return [], [], "No user files found in job"

    contents, error = resolve_job_file_args(user_files, contents)
    if error:
        return [], [], error
    file_paths, error = resolve_job_file_args(user_files, file_paths)
    if error:
        return [], [], error

    error = validate_auto_approve_job_inputs(user_files, contents, file_paths)
    if error:
        return [], [], error

    content_rel_paths, name_only = resolve_auto_approve_file_args(
        user_files, contents, file_paths
    )
    error = validate_auto_approve_object_covers_job(
        user_files, content_rel_paths, name_only
    )
    if error:
        return [], [], error
    error = validate_content_files_are_text(user_files, content_rel_paths)
    if error:
        return [], [], error
    return content_rel_paths, name_only, None


def get_job_user_files(job) -> dict[str, Path]:
    """Files of a job as {path relative to the submission root: abs_path}.

    The root holds `code/`, `run.sh` and `config.yaml`, and `run.sh` is the
    file the runner executes, so an approval built from this map pins it.

    A job that has run carries directories its run created, and the jobs this
    rule must approve are still pending and carry none, so those are left out.
    A job that has not run carries only what the submitter shipped, and a
    submitted `__pycache__` or `outputs` is in every later job of that shape:
    leaving it out would write a rule that the matcher, which counts every
    file, can never satisfy.
    """
    user_files = get_job_submission_files(job)
    if str(getattr(job, "status", "")) in NOT_RUN_STATUSES:
        return user_files
    return {
        rel: f
        for rel, f in user_files.items()
        if not set(PurePosixPath(rel).parts) & _GENERATED_DIRS
    }


# perhaps add this later again
# def authenticate(
#     credentials_path: str | Path | None = None,
# ) -> AuthResult:
#     """Set up Gmail and Drive authentication interactively.

#     Guides you through the OAuth flow step by step.
#     Works in Colab, Jupyter, and terminal environments.

#     Args:
#         credentials_path: Path to credentials.json. Defaults to ~/.syft-bg/credentials.json

#     Returns:
#         AuthResult with status of each token.

#     Example:
#         >>> import syft_bg
#         >>> syft_bg.authenticate()
#     """
#     creds_dir = get_syftbg_dir()
#     colab = is_colab()

#     creds_path = (
#         Path(credentials_path).expanduser()
#         if credentials_path
#         else creds_dir / "credentials.json"
#     )

#     if not creds_path.exists():
#         steps = credentials_setup_steps(creds_path, colab)
#         msg = (
#             f"credentials.json not found at {creds_path}\n{steps}\n"
#             "  Then run syft_bg.authenticate() again"
#         )
#         return AuthResult(success=False, error=msg)

#     gmail_out_token_path = creds_dir / "gmail_token.json"
#     drive_out_token_path = creds_dir / "drive_token.json"
#     gmail_ok = gmail_out_token_path.exists()
#     drive_ok = drive_out_token_path.exists() or colab

#     # --- Gmail token ---
#     if not gmail_ok:
#         authenticate_and_save(gmail_out_token_path, creds_path)
#     else:
#         print(f"Gmail token already exists at {gmail_out_token_path}")

#     # --- Drive token ---
#     if colab:
#         print("Drive authentication: handled natively by Colab")
#         drive_ok = True
#     elif not drive_ok:
#         authenticate_drive(drive_out_token_path, creds_path)
#     else:
#         print(f"Drive token already exists at {drive_out_token_path}")

#     # Save GCP project ID from credentials.json into config so it's
#     # available at runtime without needing the credentials file.
#     save_gcp_project_id(creds_path)

#     return AuthResult(
#         success=gmail_ok and drive_ok,
#         gmail_ok=gmail_ok,
#         drive_ok=drive_ok,
#     )
