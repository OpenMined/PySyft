import random
import io
import shutil
import tarfile
import time
from hashlib import sha256
from pathlib import Path

from syft_client.sync.environments.environment import Environment


def check_env() -> Environment:
    import os

    # Daemon subprocesses on Colab can import google.colab but cannot
    # use interactive auth (no IPython kernel), so treat them as Jupyter.
    if os.environ.get("SYFT_BG_DAEMON"):
        return Environment.JUPYTER
    try:
        import google.colab  # noqa: F401

        return Environment.COLAB
    except Exception:
        return Environment.JUPYTER


def get_email_colab() -> str | None:
    from google.colab import auth
    from googleapiclient.discovery import build

    auth.authenticate_user()

    oauth2 = build("oauth2", "v2")
    userinfo = oauth2.userinfo().get().execute()
    return userinfo.get("email")


def get_event_hash_from_content(content: str | bytes) -> str:
    # Check if content is a string (has encode method)
    if isinstance(content, str):
        return sha256(content.encode("utf-8")).hexdigest()
    # Otherwise, treat as bytes or bytes-like object
    else:
        # Ensure we have bytes for sha256
        if not isinstance(content, bytes):
            content = bytes(content)
        return sha256(content).hexdigest()


def create_event_timestamp() -> float:
    return time.time()


def random_email():
    return f"test{random.randint(1, 1000000)}@test.com"


def random_syftbox_folder_for_testing():
    return Path(f"/tmp/sb_folder_testing-{random.randint(1, 1000000)}")


def compress_data(data: bytes) -> bytes:
    tar_bytes = io.BytesIO()

    with tarfile.open(fileobj=tar_bytes, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="proposed_file_changes.json")
        info.size = len(data)
        tar.addfile(tarinfo=info, fileobj=io.BytesIO(data))
    tar_bytes.seek(0)
    compressed_data = tar_bytes.getvalue()
    return compressed_data


def uncompress_data(data: bytes) -> bytes:
    tar_bytes = io.BytesIO(data)
    with tarfile.open(fileobj=tar_bytes, mode="r:gz") as tar:
        info = tar.getmember("proposed_file_changes.json")
        data = tar.extractfile(info).read()
    return data


def _get_default_syftbox_path(email: str) -> Path:
    """Return the default local SyftBox directory for the current environment."""
    env = check_env()
    if env == Environment.COLAB:
        return Path("/content") / f"SyftBox_{email}"
    else:
        return Path.home() / f"SyftBox_{email}"


def _resolve_email(email: str | None) -> str:
    """Resolve email, auto-detecting on Colab if not provided.

    Raises:
        ValueError: If email cannot be resolved.
    """
    env = check_env()
    if env == Environment.COLAB and email is None:
        email = get_email_colab()
    if email is None:
        raise ValueError(
            "email is required when running locally. On Colab it can be auto-detected."
        )
    return email


def _resolve_token_path(token_path: str | Path | None) -> Path | None:
    """Resolve token_path from env var if not provided.

    Raises:
        ValueError: If token_path cannot be resolved outside Colab.
    """
    from syft_client.sync.config.config import settings

    env = check_env()
    if env != Environment.COLAB and token_path is None:
        resolved = settings.token_path
        if not resolved:
            raise ValueError(
                "token_path is required when running locally. "
                "Set SYFTCLIENT_TOKEN_PATH env var or pass token_path explicitly."
            )
        token_path = resolved
    return Path(token_path) if token_path is not None else None


def _delete_local_syftbox_dirs(local_syftbox_path: Path, verbose: bool = True) -> None:
    """Delete a local SyftBox directory and its companion cache directories."""
    syftbox_name = local_syftbox_path.name
    syftbox_parent = local_syftbox_path.parent
    dirs_to_delete = [
        local_syftbox_path,
        syftbox_parent / f"{syftbox_name}-events",
        syftbox_parent / f"{syftbox_name}-event-messages",
    ]
    for d in dirs_to_delete:
        if d.exists():
            shutil.rmtree(d)
            if verbose:
                print(f"Deleted local directory: {d}")


def delete_local_syftbox(
    email: str | None = None,
    local_syftbox_path: str | Path | None = None,
    verbose: bool = True,
) -> None:
    """Delete local SyftBox directories.

    Deletes the SyftBox directory and its companion cache directories
    (``<name>-events``, ``<name>-event-messages``).

    Args:
        email: Google account email. Used to resolve default path if
            ``local_syftbox_path`` is not provided.
        local_syftbox_path: Optional path to the local SyftBox directory.
            If not provided, the default path for the current environment
            is used automatically.
        verbose: If True (default), print deletion progress.
    """
    if local_syftbox_path is not None:
        resolved_path = Path(local_syftbox_path)
    else:
        if email is None:
            env = check_env()
            if env == Environment.COLAB:
                email = get_email_colab()
            if email is None:
                raise ValueError(
                    "email is required to resolve default local path. "
                    "Provide email or local_syftbox_path explicitly."
                )
        resolved_path = _get_default_syftbox_path(email)

    _delete_local_syftbox_dirs(resolved_path, verbose=verbose)


def delete_syftbox(
    token_path: str | Path | None = None,
    email: str | None = None,
    local_syftbox_path: str | Path | None = None,
    verbose: bool = True,
) -> None:
    """Delete all SyftBox state from Google Drive and local directories.

    Convenience wrapper that calls both ``delete_remote_syftbox`` and
    ``delete_local_syftbox``.

    Note: This function does NOT broadcast ``is_deleted`` events to peers.

    Args:
        token_path: Path to OAuth token JSON file. Required when running
            locally. On Colab, pass ``None`` to use Colab's built-in auth.
        email: Google account email. Required when running locally. On Colab,
            this is auto-detected if not provided.
        local_syftbox_path: Optional path to the local SyftBox directory.
            If not provided, the default path for the current environment
            is used automatically.
        verbose: If True (default), print deletion progress.
    """
    from syft_client.gdrive_utils import delete_remote_syftbox

    email_resolved = _resolve_email(email)
    token_path_resolved = _resolve_token_path(token_path)

    delete_remote_syftbox(
        token_path=token_path_resolved,
        email=email_resolved,
        verbose=verbose,
    )
    delete_local_syftbox(
        email=email_resolved,
        local_syftbox_path=local_syftbox_path,
        verbose=verbose,
    )
