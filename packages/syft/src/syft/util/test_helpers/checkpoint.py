# stdlib
import datetime
import json
import os
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

# relative
from ...service.migration.object_migration_state import MigrationData
from ...service.response import SyftSuccess
from ...service.worker.worker_image import SyftWorkerImage
from ...service.worker.worker_pool import WorkerPool
from .worker_helpers import build_and_push_image
from .worker_helpers import prune_worker_pool_and_images

CHECKPOINT_ROOT = "checkpoints"
CHECKPOINT_DIR_PREFIX = "chkpt"


def get_notebook_name_from_pytest_env() -> str | None:
    """
    Returns the notebook file name from the test environment variable 'PYTEST_CURRENT_TEST'.
    If not available, returns None.
    """
    pytest_current_test = os.environ.get("PYTEST_CURRENT_TEST", "")
    # Split by "::" and return the first part, which is the file path
    return os.path.basename(pytest_current_test.split("::")[0])


def current_nbname() -> Path:
    """Retrieve the current Jupyter notebook name."""
    curr_kernel_file = Path(ipykernel.get_connection_file())
    kernel_file = json.loads(curr_kernel_file.read_text())
    nb_name = kernel_file.get("jupyter_session", "")
    if not nb_name:
        nb_name = get_notebook_name_from_pytest_env()
    return Path(nb_name)


def root_checkpoint_path() -> Path:
    return get_root_data_path() / CHECKPOINT_ROOT


def checkpoint_parent_dir(server_uid: str, nb_name: str | None = None) -> Path:
    """Return the checkpoint directory for the current notebook and server."""
    if is_interpreter_jupyter:
        nb_name = nb_name if nb_name else current_nbname().stem
        return Path(f"{nb_name}/{server_uid}") if nb_name else Path(server_uid)
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


def get_registry_credentials() -> tuple[str, str]:
    return os.environ.get("REGISTRY_USERNAME", ""), os.environ.get(
        "REGISTRY_PASSWORD", ""
    )


def load_from_checkpoint(
    client: SyftClient,
    prev_nb_filename: str | None = None,
    root_email: str | None = None,
    root_pwd: str | None = None,
    registry_username: str | None = None,
    registry_password: str | None = None,
) -> None:
    """Load the last saved checkpoint for the given notebook state."""
    if prev_nb_filename is None:
        print("Loading from the last checkpoint of the current notebook.")
        prev_nb_filename = current_nbname().stem

    root_email = "info@openmined.org" if root_email is None else root_email
    root_pwd = "changethis" if root_pwd is None else root_pwd

    root_client = (
        client
        if is_admin(client)
        else client.login(email=root_email, password=root_pwd)
    )
    latest_checkpoint_dir = last_checkpoint_path_for_nb(
        client.id.to_string(), prev_nb_filename
    )

    if latest_checkpoint_dir is None:
        print(f"No last checkpoint found for notebook: {prev_nb_filename}")
        return

    print(f"Loading from checkpoint: {latest_checkpoint_dir}")
    result = root_client.load_migration_data(
        path_or_data=latest_checkpoint_dir / "migration.blob"
    )

    if isinstance(result, SyftError):
        raise SyftException(message=result.message)

    print("Successfully loaded data from checkpoint.")

    migration_data = MigrationData.from_file(latest_checkpoint_dir / "migration.blob")

    # klass_for_migrate_data = [
    #     WorkerPool.__canonical_name__,
    #     SyftWorkerImage.__canonical_name__,
    # ]

    # pool_and_image_data = MigrationData(
    #     server_uid=migration_data.server_uid,
    #     signing_key=migration_data.signing_key,
    #     store_objects={
    #         k: v
    #         for k, v in migration_data.store_objects.items()
    #         if k.__canonical_name__ in klass_for_migrate_data
    #     },
    #     metadata={
    #         k: v
    #         for k, v in migration_data.metadata.items()
    #         if k.__canonical_name__ not in klass_for_migrate_data
    #     },
    #     action_objects=[],
    #     blob_storage_objects=[],
    #     blobs={},
    # )

    prune_worker_pool_and_images(root_client)

    worker_images: list[SyftWorkerImage] = migration_data.get_items_by_canonical_name(
        SyftWorkerImage.__canonical_name__
    )

    # Overwrite the registry credentials if provided, else use the environment variables
    env_registry_username, env_registry_password = get_registry_credentials()
    registry_password = (
        registry_password if registry_password else env_registry_password
    )
    registry_username = (
        registry_username if registry_username else env_registry_username
    )

    old_image_to_new_image_id_map = {}
    for old_image in worker_images:
        submit_result = root_client.api.services.worker_image.submit(
            worker_config=old_image.config
        )

        assert isinstance(submit_result, SyftSuccess)

        new_image = submit_result.value

        old_image_to_new_image_id_map[old_image.id] = new_image.id

        if not new_image.is_prebuilt:
            registry_uid = (
                old_image.image_identifier.registry.id
                if old_image.image_identifier.registry
                else None
            )

            # TODO: Later add prompt support for registry credentials

            build_and_push_image(
                root_client,
                new_image,
                registry_uid=registry_uid,
                tag=old_image.image_identifier.repo_with_tag,
                reg_password=registry_username,
                reg_username=registry_password,
            )

    worker_pools: list[WorkerPool] = migration_data.get_items_by_canonical_name(
        WorkerPool.__canonical_name__
    )

    for old_pool in worker_pools:
        new_image_uid = old_image_to_new_image_id_map[old_image.id]

        root_client.worker_pools.launch(
            pool_name=old_pool.name,
            image_uid=new_image_uid,
            num_workers=old_pool.max_count,
            registry_username=registry_username,
            registry_password=registry_username,
        )

    print("Successfully loaded worker pool data from checkpoint.")
