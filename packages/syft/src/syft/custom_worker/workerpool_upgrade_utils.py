# stdlib
from pathlib import Path

# third party
from IPython.display import display

# syft absolute
import syft as sy

# relative
from ..client.client import SyftClient
from ..service.migration.object_migration_state import MigrationData
from ..service.worker.worker_image import SyftWorkerImage
from ..service.worker.worker_pool import WorkerPool
from .config import PrebuiltWorkerConfig


def upgrade_prebuilt_image(
    client: SyftClient, pool: WorkerPool, old_image: SyftWorkerImage
) -> SyftWorkerImage | None:
    print(f"Found outdated prebuilt worker image with config `{old_image.config}`\n")
    new_image_tag = input(
        "Please enter the tag for the upgraded image. Type 'skip' to skip upgrading this workerpool"
    )
    if new_image_tag.lower() == "skip":
        return None

    new_config = sy.PrebuiltWorkerConfig(
        tag=new_image_tag, description=old_image.config.description
    )
    confirmation = input(
        f"Created new config `{new_config}`. is this correct? (y/n)"
    ).lower()
    if confirmation == "y":
        print("submitting image...")
        try:
            result = client.api.services.worker_image.submit(worker_config=new_config)
            display(result)
            return result.value
        except Exception as e:
            display(e)
            print("starting over")
            return upgrade_prebuilt_image(client, pool, old_image)
    else:
        print("starting over")
        return upgrade_prebuilt_image(client, pool, old_image)


def upgrade_syft_image(
    client: SyftClient, pool: WorkerPool, old_image: SyftWorkerImage
) -> SyftWorkerImage | None:
    # TODO
    print(f"Skipping image {old_image.image_identifier}, this is not a pre-built image")
    return None


def upgrade_workerpool(
    client: SyftClient,
    pool: WorkerPool,
    migration_data: MigrationData,
) -> bool:
    if pool.name == migration_data.default_pool_name:
        print("Skipping default pool, this pool has already been upgraded")
        return False

    print(f"Upgrading workerpool {pool.name}")

    images = migration_data.get_items_by_canonical_name(
        SyftWorkerImage.__canonical_name__
    )
    image_id = pool.image_id
    old_image = [img for img in images if img.id == image_id][0]
    is_prebuilt_image = isinstance(old_image.config, PrebuiltWorkerConfig)

    if is_prebuilt_image:
        new_image = upgrade_prebuilt_image(client, pool, old_image)
        if not new_image:
            print(f"Skipping workerpool {pool.name}.")
            return False
    else:
        # TODO upgrade prebuilt images
        new_image = upgrade_syft_image(client, pool, old_image)

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


def upgrade_custom_workerpools(
    client: SyftClient, migration_data: str | Path | MigrationData
) -> None:
    print("This is a utility to upgrade workerpools to the new syft version")
    print("If an upgrade fails, it is always possible to start the workerpool manually")

    if isinstance(migration_data, str | Path):
        print("loading migration data...")
        migration_data = MigrationData.from_file(migration_data)

    worker_pools = migration_data.get_items_by_canonical_name(
        WorkerPool.__canonical_name__
    )
    num_upgraded = 0
    for pool in worker_pools:
        is_upgraded = upgrade_workerpool(client, pool, migration_data)
        if is_upgraded:
            num_upgraded += 1
        print()

    print(f"Upgraded {num_upgraded} workerpools to the new syft version")
    print("Please verify your upgraded pools with `client.worker_pools`")
