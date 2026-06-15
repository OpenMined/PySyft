"""Utility functions used by the syft-bg API layer."""

import hashlib
import shutil
from pathlib import Path

from syft_bg.approve.config import AutoApproveConfig, FileEntry
from syft_bg.common.config import get_syftbg_dir, get_default_paths
from syft_bg.common.drive import is_colab
from syft_bg.common.syft_bg_config import SyftBgConfig
from syft_bg.email_approve.pubsub_setup import get_project_id_from_credentials


from syft_bg.services.base import ServiceInfo, ServiceStatus

PERMISSION_FILE_NAME = "syft.pub.yaml"
DEFAULT_NAME_ONLY_FILES = {"params.json"}


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
        config = SyftBgConfig.from_path()
        config.email_approve.gcp_project_id = project_id
        config.save()
    except Exception:
        pass


def generate_unique_name(
    name: str | None,
    content_files: list[tuple[str, Path]],
    config: AutoApproveConfig,
) -> str:
    """Generate a unique name for an auto-approval object."""
    if name is None:
        if content_files:
            first_rel = content_files[0][0]
            name = Path(first_rel).stem if len(content_files) == 1 else "auto_approval"
        else:
            name = "auto_approval"

    if name in config.auto_approvals.objects:
        base_name = name
        counter = 1
        while f"{base_name}_{counter}" in config.auto_approvals.objects:
            counter += 1
        name = f"{base_name}_{counter}"

    return name


def resolve_content_files(
    contents: list[str | Path], base_dir: Path | None
) -> tuple[list[tuple[str, Path]], str | None]:
    """Resolve content paths to (relative_path, absolute_path) pairs.

    Returns (content_files, error). error is None on success.
    """
    content_files: list[tuple[str, Path]] = []
    for item in contents:
        if base_dir is not None:
            rel = str(item)
            abs_path = base_dir / rel
            if not abs_path.exists():
                return [], f"File not found: {abs_path}"
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


def copy_and_hash_files(
    content_files: list[tuple[str, Path]], name: str
) -> list[FileEntry]:
    """Copy files to the managed auto-approvals directory and compute hashes."""
    obj_dir = get_default_paths().auto_approvals_dir / name
    obj_dir.mkdir(parents=True, exist_ok=True)

    entries: list[FileEntry] = []
    for rel_path, abs_path in content_files:
        dest = obj_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(abs_path, dest)
        content = dest.read_text(encoding="utf-8")
        file_hash = "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
        entries.append(
            FileEntry(relative_path=rel_path, path=str(dest), hash=file_hash)
        )
    return entries


def resolve_auto_approve_file_args(
    user_files: dict[str, Path],
    contents: list[str] | None,
    file_paths: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Determine which job files are content-matched vs name-only.

    Returns (content_rel_paths, name_only).
    """
    if contents is None and file_paths is None:
        all_files = set(user_files.keys())
        name_only = all_files.intersection(DEFAULT_NAME_ONLY_FILES)
        content_matched = all_files - name_only
        return list(content_matched), list(name_only)
    elif contents is not None and file_paths is None:
        return list(contents), []
    elif contents is None and file_paths is not None:
        name_only = list(file_paths)
        content_rel_paths = set(user_files.keys()) - set(file_paths)
        return list(content_rel_paths), name_only
    else:
        return list(contents), list(file_paths)  # type: ignore[arg-type]


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


_GENERATED_DIRS = {".venv", "outputs", "__pycache__"}


def get_job_user_files(job) -> dict[str, Path]:
    """Get user files from a job's code directory as {relative_path: abs_path} mapping."""
    user_files: dict[str, Path] = {}
    code_dir = job.code_dir
    if code_dir.exists():
        for f in code_dir.rglob("*"):
            if not f.is_file() or f.name == PERMISSION_FILE_NAME:
                continue
            # Skip files inside directories generated during job execution
            rel = f.relative_to(code_dir)
            if rel.parts[0] in _GENERATED_DIRS:
                continue
            user_files[str(rel)] = f
    return user_files


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
