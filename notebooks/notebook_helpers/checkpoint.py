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

NB_STATE_DIRECTORY = "nb_checkpoints"


def current_nbname() -> Path:
    """Get the current notebook name"""
    curr_kernel_file = Path(ipykernel.get_connection_file())
    kernel_file = json.loads(curr_kernel_file.read_text())
    nb_name = kernel_file["jupyter_session"]
    return Path(nb_name)


def get_or_create_state_dir(filename: str) -> Path:
    """Get or create the state directory for the given filename."""

    # Generate the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"checkpoint_{timestamp}"

    nb_path = get_root_data_path() / NB_STATE_DIRECTORY
    filepath = nb_path / filename / checkpoint_dir
    if not filepath.exists():
        filepath.mkdir(parents=True, exist_ok=True)
    return filepath


def state_dir_for_nb(server_uid: str) -> Path:
    """State directory for the current notebook"""
    nb_name = current_nbname().stem  # Get the filename without extension
    return get_or_create_state_dir(filename=f"{nb_name}/{server_uid}")


def is_admin(client: SyftClient) -> bool:
    """Is User an admin."""
    return client._SyftClient__user_role == ServiceRole.ADMIN


def checkpoint_db(
    client: SyftClient,
    root_email: str = "info@openmined.org",
    root_pwd: str = "changethis",
) -> None:
    """Save checkpoint for database."""

    # Get root client (login if not already admin)
    root_client = (
        client
        if is_admin(client)
        else client.login(email=root_email, password=root_pwd)
    )

    # Get migration data from the database
    migration_data = root_client.get_migration_data(include_blobs=True)
    if isinstance(migration_data, SyftError):
        raise SyftException(message=migration_data.message)

    # Define the state directory for the current notebook and server
    state_dir = state_dir_for_nb(server_uid=client.id.to_string())

    # Save migration data in blob and yaml format
    migration_data.save(
        path=state_dir / "migration.blob", yaml_path=state_dir / "migration.yaml"
    )

    print(f"Checkpoint saved at: \n {state_dir}")
    return state_dir


def last_db_checkpoint_dir(filename: str, server_id: str) -> Path | None:
    """Return the directory of the latest checkpoint for the given filename."""

    filename = filename.split(".json")[0]
    checkpoint_parent_dir = get_or_create_state_dir(f"{filename}/{server_id}").parent

    # Get all checkpoint directory matching the pattern
    checkpoint_dirs = [
        d for d in checkpoint_parent_dir.glob("checkpoint_*") if d.is_dir()
    ]

    checkpoints_dirs_with_blob_entry = [
        d for d in checkpoint_dirs if any(d.glob("*.blob"))
    ]

    if checkpoints_dirs_with_blob_entry:
        # Return the latest directory based on modification time
        return max(checkpoints_dirs_with_blob_entry, key=lambda d: d.stat().st_mtime)

    return None


def load_from_checkpoint(
    prev_nb_filename: str,
    client: SyftClient,
    root_email: str = "info@openmined.org",
    root_pwd: str = "changethis",
) -> None:
    """Load the last saved checkpoint for the given notebook state."""

    # Get root client (login if not already admin)
    root_client = (
        client
        if is_admin(client)
        else client.login(email=root_email, password=root_pwd)
    )

    lastest_checkpoint_dir = last_db_checkpoint_dir(
        prev_nb_filename, client.id.to_string()
    )

    if lastest_checkpoint_dir is None:
        print("No previous checkpoint found !")
        return

    print(f"Loading from checkpoint: {lastest_checkpoint_dir}")

    result = root_client.load_migration_data(
        path=lastest_checkpoint_dir / "migration.blob"
    )

    if isinstance(result, SyftError):
        raise SyftException(message=result.message)

    print("Successfully loaded data from checkpoint.")
