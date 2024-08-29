# stdlib
from pathlib import Path
from typing import cast

# third party
from IPython.display import display

# syft absolute
import syft as sy

# relative
from ..client.client import SyftClient
from ..service.migration.object_migration_state import MigrationData
from ..service.worker.image_identifier import SyftWorkerImageIdentifier
from ..service.worker.worker_image import SyftWorkerImage
from ..service.worker.worker_pool import WorkerPool


def upgrade_custom_workerpools(
    client: SyftClient,
    migration_data: str | Path | MigrationData,
    mode: str = "manual",
) -> None:
    """Upgrade custom workerpools to the new syft version

    Args:
        client (SyftClient): Admin client to upgrade workerpools with
        migration_data (str | Path | MigrationData): Path to migration data or MigrationData object
        mode (str, optional): if "auto" the upgrade will be done automatically. "auto" assumes
            all images and tags use Syft versioning. Defaults to "manual".

    Raises:
        ValueError: if mode is not "manual" or "auto"
    """
    print("This is a utility to upgrade workerpools to the new syft version")
    print("If an upgrade fails, it is always possible to start the workerpool manually")

    if mode not in ["manual", "auto"]:
        raise ValueError("mode must be either 'manual' or 'auto'")

    if isinstance(migration_data, str | Path):
        print("loading migration data...")
        migration_data = MigrationData.from_file(migration_data)

    # mypy does not recognize instance check for str | Path
    migration_data = cast(MigrationData, migration_data)
    worker_pools = migration_data.get_items_by_canonical_name(
        WorkerPool.__canonical_name__
    )
    num_upgraded = 0
    for pool in worker_pools:
        is_upgraded = upgrade_workerpool(client, pool, migration_data, mode)
        if is_upgraded:
            num_upgraded += 1
        print()

    print(f"Upgraded {num_upgraded} workerpools to the new syft version")
    print("Please verify your upgraded pools with `client.worker_pools`")


def upgrade_workerpool(
    client: SyftClient,
    pool: WorkerPool,
    migration_data: MigrationData,
    mode: str = "manual",
) -> bool:
    if pool.name == migration_data.default_pool_name:
        print("Skipping default pool, this pool has already been upgraded")
        return False

    print(f"Upgrading workerpool {pool.name}")

    images = migration_data.get_items_by_canonical_name(
        SyftWorkerImage.__canonical_name__
    )
    image_id = pool.image_id
    old_image: SyftWorkerImage = [img for img in images if img.id == image_id][0]

    if old_image.is_prebuilt:
        new_image = upgrade_prebuilt_image(client, old_image, mode)
    else:
        new_image = upgrade_syft_image(client, old_image, mode)

    if not new_image:
        print(f"Failed to upgrade workerpool {pool.name}, could not build new image")
        return False

    print(f"starting new pool `{pool.name}` with {pool.max_count} workers")
    try:
        result = client.api.services.worker_pool.launch(
            pool_name=pool.name,
            image_uid=new_image.id,
            num_workers=pool.max_count,
        )
        display(result)
        return True
    except Exception as e:
        display(e)
        print(f"failed to start workerpool {pool.name}, please start the pool manually")
        return False


def upgrade_prebuilt_image(
    client: SyftClient,
    old_image: SyftWorkerImage,
    mode: str = "manual",
) -> SyftWorkerImage | None:
    print(f"Found outdated prebuilt worker image `{old_image.image_identifier}`")
    if mode == "auto":
        new_syft_version = client.metadata.syft_version  # type: ignore
        new_identifier = upgrade_image_identifier(
            old_image.image_identifier, new_syft_version
        )
        new_image_tag = new_identifier.full_name_with_tag
    else:
        new_image_tag_or_none = get_tag_from_input()
        if not new_image_tag_or_none:
            return None
        new_image_tag = new_image_tag_or_none

    new_config = sy.PrebuiltWorkerConfig(
        tag=new_image_tag, description=old_image.config.description
    )

    print("submitting new prebuilt image...")
    try:
        result = client.api.services.worker_image.submit(worker_config=new_config)
        display(result)
        return result.value
    except Exception as e:
        print("could not submit new image")
        display(e)
        return None


def upgrade_syft_image(
    client: SyftClient,
    old_image: SyftWorkerImage,
    mode: str = "manual",
) -> SyftWorkerImage | None:
    old_identifier = old_image.image_identifier
    old_config = old_image.config
    new_syft_version = client.metadata.syft_version  # type: ignore

    if old_identifier is None:
        raise ValueError("old image does not have an image identifier")

    print(f"Found outdated custom worker image `{old_image.image_identifier}`")

    new_dockerfile = update_dockerfile_baseimage_tag(
        old_config.dockerfile, new_syft_version
    )

    if mode == "manual":
        confirm = confirm_dockerfile_update(old_config.dockerfile, new_dockerfile)
        if not confirm:
            return None

    # NOTE do not copy filename, it does not match the new dockerfile
    new_config = sy.DockerWorkerConfig(
        dockerfile=new_dockerfile, description=old_config.description, file_name=None
    )
    new_identifier = upgrade_image_identifier(old_identifier, new_syft_version)
    print(
        f"Updating image tag from {old_identifier.repo_with_tag} to {new_identifier.repo_with_tag}"
    )

    print("submitting new image...")
    try:
        submit_result = client.api.services.worker_image.submit(
            worker_config=new_config
        )
        custom_image = submit_result.value
    except Exception as e:
        print("could not submit new image")
        display(e)
        return None

    print("building new image...")
    try:
        client.api.services.worker_image.build(
            image_uid=custom_image.id,
            tag=new_identifier.repo_with_tag,
        )
    except Exception as e:
        print("could not build new image")
        display(e)
        return None

    return custom_image


def get_tag_from_input() -> str | None:
    new_image_tag = input(
        "Please enter the tag for the upgraded image. Type 'skip' to skip upgrading this workerpool"
    )
    if new_image_tag.lower() == "skip":
        return None
    return new_image_tag


def update_dockerfile_baseimage_tag(old_dockerfile: str, new_tag: str) -> str:
    is_updated = False
    new_dockerfile_ = []
    for line in old_dockerfile.splitlines():
        if line.startswith("FROM openmined/syft-backend:"):
            updated_line = f"FROM openmined/syft-backend:{new_tag}"
            new_dockerfile_.append(updated_line)
            is_updated = True
        else:
            new_dockerfile_.append(line)

    if not is_updated:
        raise ValueError("Could not update baseimage")
    return "\n".join(new_dockerfile_)


def confirm_dockerfile_update(old_dockerfile: str, new_dockerfile: str) -> bool:
    print("updated your dockerfile baseimage:")
    print("- Old dockerfile ----")
    print(old_dockerfile)
    print("- New dockerfile ----")
    print(new_dockerfile)
    print("---------------------")
    confirmation = input("is this correct? [y/n]")
    if confirmation.lower() not in ["y", "n"]:
        return confirm_dockerfile_update(old_dockerfile, new_dockerfile)
    return confirmation.lower() == "y"


def upgrade_image_identifier(
    old_identifier: SyftWorkerImageIdentifier, new_tag: str
) -> SyftWorkerImageIdentifier:
    return SyftWorkerImageIdentifier(
        registry=old_identifier.registry, repo=old_identifier.repo, tag=new_tag
    )
