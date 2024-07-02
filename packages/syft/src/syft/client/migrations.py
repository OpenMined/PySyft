# stdlib
from io import BytesIO
import sys

# relative
from ..serde.serialize import _serialize
from ..service.response import SyftError
from ..service.response import SyftSuccess
from ..types.blob_storage import BlobStorageEntry
from ..types.blob_storage import CreateBlobStorageEntry
from ..types.syft_object import Context
from ..types.syft_object import SyftObject
from .domain_client import DomainClient


def migrate_blob_storage_object(
    from_client: DomainClient,
    to_client: DomainClient,
    obj: SyftObject,
) -> SyftSuccess | SyftError:
    migrated_obj = obj.migrate_to(BlobStorageEntry.__version__, Context())
    uploaded_by = migrated_obj.uploaded_by
    blob_retrieval = from_client.services.blob_storage.read(obj.id)
    if isinstance(blob_retrieval, SyftError):
        return blob_retrieval

    data = blob_retrieval.read()
    serialized = _serialize(data, to_bytes=True)
    size = sys.getsizeof(serialized)
    blob_create = CreateBlobStorageEntry.from_blob_storage_entry(obj)
    blob_create.file_size = size

    blob_deposit_object = to_client.services.blob_storage.allocate_for_user(
        blob_create, uploaded_by
    )
    if isinstance(blob_deposit_object, SyftError):
        return blob_deposit_object
    return blob_deposit_object.write(BytesIO(serialized))


def migrate_blob_storage(
    from_client: DomainClient,
    to_client: DomainClient,
    blob_storage_objects: list[SyftObject],
) -> SyftSuccess | SyftError:
    for obj in blob_storage_objects:
        migration_result = migrate_blob_storage_object(from_client, to_client, obj)
        if isinstance(migration_result, SyftError):
            return migration_result
    return SyftSuccess(message="Blob storage migration successful.")


def migrate(
    from_client: DomainClient, to_client: DomainClient
) -> SyftSuccess | SyftError:
    migration_data = from_client.get_migration_data()
    if isinstance(migration_data, SyftError):
        return migration_data

    # Blob storage is migrated via client
    blob_storage_objects = migration_data.blob_storage_objects
    blob_migration_result = migrate_blob_storage(
        from_client, to_client, blob_storage_objects
    )
    if isinstance(blob_migration_result, SyftError):
        return blob_migration_result

    # Rest of the migration data is migrated via service
    return to_client.api.services.migration.apply_migration_data(migration_data)
