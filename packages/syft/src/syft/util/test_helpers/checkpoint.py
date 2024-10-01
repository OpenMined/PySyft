# stdlib
import datetime
import os
from pathlib import Path

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


def root_checkpoint_path() -> Path:
    return get_root_data_path() / CHECKPOINT_ROOT


def get_checkpoint_parent_dir(server_uid: str, chkpt_name: str) -> Path:
    return root_checkpoint_path() / chkpt_name / server_uid


def create_checkpoint_dir(server_uid: str, chkpt_name: str) -> Path:
    """Create a checkpoint directory by chkpt_name and server_uid."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"{CHECKPOINT_DIR_PREFIX}_{timestamp}"
    checkpoint_parent_dir = get_checkpoint_parent_dir(
        server_uid=server_uid, chkpt_name=chkpt_name
    )
    checkpoint_full_path = checkpoint_parent_dir / checkpoint_dir

    # Format of Checkpoint Directory:
    #  <root_syft_dir>/checkpoints/chkpt_name/<server_uid>/chkpt_<timestamp>

    checkpoint_full_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_full_path


def is_admin(client: SyftClient) -> bool:
    return client._SyftClient__user_role == ServiceRole.ADMIN


def create_checkpoint(
    name: str,  # Name of the checkpoint
    client: SyftClient,
    root_email: str | None = None,
    root_pwd: str | None = None,
) -> None:
    """Save a checkpoint for the database."""

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

    checkpoint_dir = create_checkpoint_dir(
        server_uid=client.id.to_string(), chkpt_name=name
    )
    migration_data.save(
        path=checkpoint_dir / "migration.blob",
        yaml_path=checkpoint_dir / "migration.yaml",
    )
    print(f"Checkpoint saved at: \n {checkpoint_dir}")


def last_checkpoint_path_for(server_uid: str, chkpt_name: str) -> Path | None:
    """Return the directory of the latest checkpoint for the given name."""

    checkpoint_parent_dir = get_checkpoint_parent_dir(
        server_uid=server_uid, chkpt_name=chkpt_name
    )

    checkpoint_dirs = [
        d
        for d in checkpoint_parent_dir.glob(f"{CHECKPOINT_DIR_PREFIX}_*")
        if d.is_dir()
    ]
    checkpoints_dirs_with_blob_entry = [
        d for d in checkpoint_dirs if any(d.glob("*.blob"))
    ]

    if checkpoints_dirs_with_blob_entry:
        print(f"Loading from the last checkpoint for: {chkpt_name}")
        return max(checkpoints_dirs_with_blob_entry, key=lambda d: d.stat().st_mtime)

    return None


def get_registry_credentials() -> tuple[str, str]:
    return os.environ.get("REGISTRY_USERNAME", ""), os.environ.get(
        "REGISTRY_PASSWORD", ""
    )


def load_from_checkpoint(
    client: SyftClient,
    name: str,
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
    latest_checkpoint_dir = last_checkpoint_path_for(
        server_uid=client.id.to_string(), chkpt_name=name
    )

    if latest_checkpoint_dir is None:
        print(f"No last checkpoint found for : {name}")
        return

    print(f"Loading from checkpoint: {latest_checkpoint_dir}")
    result = root_client.load_migration_data(
        path_or_data=latest_checkpoint_dir / "migration.blob",
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
