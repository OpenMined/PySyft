# stdlib
import os
from pathlib import Path
import tempfile
import zipfile

# syft absolute
from syft import SyftError
from syft import SyftException
from syft.client.client import SyftClient
from syft.service.user.user_roles import ServiceRole
from syft.util.util import get_root_data_path

# relative
from ...server.env import get_default_root_email
from ...server.env import get_default_root_password
from .worker_helpers import build_and_push_image

CHECKPOINT_ROOT = "checkpoints"
CHECKPOINT_DIR_PREFIX = "chkpt"
DEFAULT_CHECKPOINT_DIR = get_root_data_path() / CHECKPOINT_ROOT

try:
    # Ensure the default checkpoint path exists always
    DEFAULT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"Error creating default checkpoint directory: {e}")


def is_admin(client: SyftClient) -> bool:
    return client._SyftClient__user_role == ServiceRole.ADMIN


def is_valid_dir(path: Path | str) -> Path:
    if isinstance(path, str):
        path = Path(path)
    if not path.is_dir():
        raise SyftException(f"Path {path} is not a directory.")
    return path


def is_valid_file(path: Path | str) -> Path:
    if isinstance(path, str):
        path = Path(path)
    if not path.is_file():
        raise SyftException(f"Path {path} is not a file.")
    return path


def create_checkpoint(
    name: str,  # Name of the checkpoint
    client: SyftClient,
    chkpt_dir: Path | str = DEFAULT_CHECKPOINT_DIR,
    root_email: str | None = None,
    root_pwd: str | None = None,
) -> None:
    """Save a checkpoint for the database."""

    is_valid_dir(chkpt_dir)

    if root_email is None:
        root_email = get_default_root_email()

    if root_pwd is None:
        root_pwd = get_default_root_password()

    root_client = (
        client
        if is_admin(client)
        else client.login(email=root_email, password=root_pwd)
    )
    migration_data = root_client.get_migration_data(include_blobs=True)

    if isinstance(migration_data, SyftError):
        raise SyftException(message=migration_data.message)

    checkpoint_path = chkpt_dir / f"{name}.zip"

    # get a temporary directory to save the checkpoint
    temp_dir = Path(tempfile.mkdtemp())
    checkpoint_blob = temp_dir / "checkpoint.blob"
    checkpoint_yaml = temp_dir / "checkpoint.yaml"
    migration_data.save(
        path=checkpoint_blob,
        yaml_path=checkpoint_yaml,
    )

    # Combine the files into a single zip file to checkpoint_path
    with zipfile.ZipFile(checkpoint_path, "w") as zipf:
        zipf.write(checkpoint_blob, "checkpoint.blob")
        zipf.write(checkpoint_yaml, "checkpoint.yaml")

    print(f"Checkpoint saved at: \n {checkpoint_path}")


def get_checkpoint_for(
    path: Path | str | None = None, chkpt_name: str | None = None
) -> Path | None:
    # Path takes precedence over name
    if path:
        return is_valid_file(path)

    if chkpt_name:
        return is_valid_file(DEFAULT_CHECKPOINT_DIR / f"{chkpt_name}.zip")


def get_registry_credentials() -> tuple[str, str]:
    return os.environ.get("REGISTRY_USERNAME", ""), os.environ.get(
        "REGISTRY_PASSWORD", ""
    )


def load_from_checkpoint(
    client: SyftClient,
    name: str | None = None,
    path: Path | str | None = None,
    root_email: str | None = None,
    root_password: str | None = None,
    registry_username: str | None = None,
    registry_password: str | None = None,
) -> None:
    """Load the last saved checkpoint for the given checkpoint state."""

    root_email = "info@openmined.org" if root_email is None else root_email
    root_password = "changethis" if root_password is None else root_password

    root_client = (
        client
        if is_admin(client)
        else client.login(email=root_email, password=root_password)
    )
    if name is None and path is None:
        raise SyftException("Please provide either a checkpoint name or a path.")

    checkpoint_zip_path = get_checkpoint_for(path=path, chkpt_name=name)

    if checkpoint_zip_path is None:
        print(f"No last checkpoint found for : {name} or {path}")
        return

    # Unzip the checkpoint zip file
    with zipfile.ZipFile(checkpoint_zip_path, "r") as zipf:
        checkpoint_temp_dir = Path(tempfile.mkdtemp())
        zipf.extract("checkpoint.blob", checkpoint_temp_dir)
        zipf.extract("checkpoint.yaml", checkpoint_temp_dir)

    checkpoint_blob = checkpoint_temp_dir / "checkpoint.blob"

    print(f"Loading from checkpoint: {checkpoint_zip_path}")
    result = root_client.load_migration_data(
        path_or_data=checkpoint_blob,
        include_worker_pools=True,
        with_reset_db=True,
    )

    if isinstance(result, SyftError):
        raise SyftException(message=result.message)

    print("Successfully loaded data from checkpoint.")

    # Step 1: Build and push the worker images

    print("Recreating worker images from checkpoint.")
    worker_image_list = (
        [] if root_client.images.get_all() is None else root_client.images.get_all()
    )
    for worker_image in worker_image_list:
        if worker_image.is_prebuilt:
            continue

        registry = worker_image.image_identifier.registry

        build_and_push_image(
            root_client,
            worker_image,
            registry_uid=registry.id if registry else None,
            tag=worker_image.image_identifier.repo_with_tag,
            reg_password=registry_username,
            reg_username=registry_password,
            force_build=True,
        )

    print("Successfully Built worker image data from checkpoint.")

    # Step 2: Recreate the worker pools
    print("Recreating worker pools from checkpoint.")
    worker_pool_list = (
        [] if root_client.worker_pools is None else root_client.worker_pools
    )
    for worker_pool in worker_pool_list:
        previous_worker_cnt = worker_pool.max_count
        purge_res = root_client.worker_pools.purge_workers(pool_id=worker_pool.id)
        print(purge_res)
        add_res = root_client.worker_pools.add_workers(
            number=previous_worker_cnt,
            pool_id=worker_pool.id,
            registry_username=registry_username,
            registry_password=registry_password,
        )
        print(add_res)

    print("Successfully loaded worker pool data from checkpoint.")
