# stdlib
import datetime
import json
from pathlib import Path

# third party
import ipykernel

# syft absolute
from syft import SyftError
from syft import SyftException
from syft.client.client import SyftClient
from syft.service.user.user_roles import ServiceRole
from syft.util.util import get_root_data_path
from syft.util.util import is_interpreter_jupyter

CHECKPOINT_ROOT = "checkpoints"
CHECKPOINT_DIR_PREFIX = "chkpt"


def current_nbname() -> Path:
    """Retrieve the current Jupyter notebook name."""
    curr_kernel_file = Path(ipykernel.get_connection_file())
    kernel_file = json.loads(curr_kernel_file.read_text())
    nb_name = kernel_file["jupyter_session"]
    return Path(nb_name)


def root_checkpoint_path() -> Path:
    return get_root_data_path() / CHECKPOINT_ROOT


def checkpoint_parent_dir(server_uid: str, nb_name: str | None = None) -> Path:
    """Return the checkpoint directory for the current notebook and server."""
    if is_interpreter_jupyter:
        nb_name = nb_name if nb_name else current_nbname().stem
        return Path(f"{nb_name}/{server_uid}")
    return Path(server_uid)


def get_checkpoints_dir(server_uid: str, nb_name: str) -> Path:
    return root_checkpoint_path() / checkpoint_parent_dir(server_uid, nb_name)


def get_checkpoint_dir(
    server_uid: str, checkpoint_dir: str, nb_name: str | None = None
) -> Path:
    return get_checkpoints_dir(server_uid, nb_name) / checkpoint_dir


def create_checkpoint_dir(server_uid: str) -> Path:
    """Create a checkpoint directory for the current notebook and server."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"{CHECKPOINT_DIR_PREFIX}_{timestamp}"
    checkpoint_full_path = get_checkpoint_dir(server_uid, checkpoint_dir=checkpoint_dir)

    checkpoint_full_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_full_path


def is_admin(client: SyftClient) -> bool:
    return client._SyftClient__user_role == ServiceRole.ADMIN


def create_checkpoint(
    client: SyftClient,
    root_email: str = "info@openmined.org",
    root_pwd: str = "changethis",
) -> None:
    """Save a checkpoint for the database."""
    root_client = (
        client
        if is_admin(client)
        else client.login(email=root_email, password=root_pwd)
    )
    migration_data = root_client.get_migration_data(include_blobs=True)

    if isinstance(migration_data, SyftError):
        raise SyftException(message=migration_data.message)

    if not is_interpreter_jupyter():
        raise SyftException(
            message="Checkpoint can only be created in Jupyter Notebook."
        )

    checkpoint_dir = create_checkpoint_dir(server_uid=client.id.to_string())
    migration_data.save(
        path=checkpoint_dir / "migration.blob",
        yaml_path=checkpoint_dir / "migration.yaml",
    )
    print(f"Checkpoint saved at: \n {checkpoint_dir}")


def last_checkpoint_path_for_nb(server_uid: str, nb_name: str = None) -> Path | None:
    """Return the directory of the latest checkpoint for the given notebook."""
    nb_name = nb_name if nb_name else current_nbname().stem
    filename = nb_name.split(".ipynb")[0]
    checkpoint_parent_dir = get_checkpoints_dir(server_uid, filename)
    checkpoint_dirs = [
        d
        for d in checkpoint_parent_dir.glob(f"{CHECKPOINT_DIR_PREFIX}_*")
        if d.is_dir()
    ]
    checkpoints_dirs_with_blob_entry = [
        d for d in checkpoint_dirs if any(d.glob("*.blob"))
    ]

    if checkpoints_dirs_with_blob_entry:
        return max(checkpoints_dirs_with_blob_entry, key=lambda d: d.stat().st_mtime)

    return None


def load_from_checkpoint(
    prev_nb_filename: str,
    client: SyftClient,
    root_email: str = "info@openmined.org",
    root_pwd: str = "changethis",
) -> None:
    """Load the last saved checkpoint for the given notebook state."""
    root_client = (
        client
        if is_admin(client)
        else client.login(email=root_email, password=root_pwd)
    )
    latest_checkpoint_dir = last_checkpoint_path_for_nb(
        client.id.to_string(), prev_nb_filename
    )

    if latest_checkpoint_dir is None:
        print("No previous checkpoint found!")
        return

    print(f"Loading from checkpoint: {latest_checkpoint_dir}")
    result = root_client.load_migration_data(
        path=latest_checkpoint_dir / "migration.blob"
    )

    if isinstance(result, SyftError):
        raise SyftException(message=result.message)

    print("Successfully loaded data from checkpoint.")
