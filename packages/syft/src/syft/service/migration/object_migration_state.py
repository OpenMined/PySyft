# stdlib
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
import sys
from typing import Any

# third party
from typing_extensions import Self
import yaml

# relative
from ...serde.deserialize import _deserialize
from ...serde.serializable import serializable
from ...serde.serialize import _serialize
from ...server.credentials import SyftSigningKey
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...store.document_store_errors import NotFoundException
from ...types.blob_storage import BlobStorageEntry
from ...types.blob_storage import CreateBlobStorageEntry
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.syft_migration import migrate
from ...types.syft_object import Context
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SyftBaseObject
from ...types.syft_object import SyftObject
from ...types.syft_object_registry import SyftObjectRegistry
from ...types.transforms import make_set_default
from ...types.uid import UID
from ...util.util import prompt_warning_message
from ..response import SyftSuccess
from ..worker.utils import DEFAULT_WORKER_POOL_NAME
from ..worker.worker_image import SyftWorkerImage
from ..worker.worker_pool import SyftWorker
from ..worker.worker_pool import WorkerPool


@serializable()
class SyftObjectMigrationState(SyftObject):
    __canonical_name__ = "SyftObjectMigrationState"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_unique__ = ["canonical_name"]

    canonical_name: str
    current_version: int

    @property
    def latest_version(self) -> int | None:
        available_versions = SyftObjectRegistry.get_versions(
            canonical_name=self.canonical_name,
        )
        if not available_versions:
            return None

        return sorted(available_versions, reverse=True)[0]

    @property
    def supported_versions(self) -> list:
        return SyftObjectRegistry.get_versions(self.canonical_name)


@serializable(canonical_name="SyftMigrationStateSQLStash", version=1)
class SyftMigrationStateStash(ObjectStash[SyftObjectMigrationState]):
    @as_result(SyftException, NotFoundException)
    def get_by_name(
        self, canonical_name: str, credentials: SyftVerifyKey
    ) -> SyftObjectMigrationState:
        return self.get_one(
            credentials=credentials,
            filters={"canonical_name": canonical_name},
        ).unwrap()


@serializable()
class StoreMetadata(SyftBaseObject):
    __canonical_name__ = "StoreMetadata"
    __version__ = SYFT_OBJECT_VERSION_1

    object_type: type
    permissions: dict[UID, set[str]]
    storage_permissions: dict[UID, set[UID]]


@serializable()
class MigrationData(SyftObject):
    __canonical_name__ = "MigrationData"
    __version__ = SYFT_OBJECT_VERSION_2
    syft_version: str = ""
    default_pool_name: str = DEFAULT_WORKER_POOL_NAME
    server_uid: UID
    signing_key: SyftSigningKey
    store_objects: dict[type[SyftObject], list[SyftObject]]
    metadata: dict[type[SyftObject], StoreMetadata]
    action_objects: dict[type[SyftObject], list[SyftObject]]
    blob_storage_objects: list[SyftObject]
    blobs: dict[UID, Any] = {}

    __repr_attrs__ = [
        "server_uid",
        "root_verify_key",
        "num_objects",
        "num_action_objects",
        "includes_blobs",
    ]

    @property
    def root_verify_key(self) -> SyftVerifyKey:
        return self.signing_key.verify_key

    @property
    def num_objects(self) -> int:
        return sum(len(objs) for objs in self.store_objects.values())

    @property
    def num_action_objects(self) -> int:
        return sum(len(objs) for objs in self.action_objects.values())

    @property
    def includes_blobs(self) -> bool:
        blob_ids = [obj.id for obj in self.blob_storage_objects]
        return set(self.blobs.keys()) == set(blob_ids)

    @property
    def includes_custom_workerpools(self) -> bool:
        cname = WorkerPool.__canonical_name__
        worker_pools = None
        for k, v in self.store_objects.items():
            if k.__canonical_name__ == cname:
                worker_pools = v

        if worker_pools is None:
            return False

        custom_pools = [
            pool
            for pool in worker_pools
            if getattr(pool, "name", None) != self.default_pool_name
        ]
        return len(custom_pools) > 0

    def make_migration_config(self) -> dict[str, Any]:
        server_uid = self.server_uid.to_string()
        server_private_key = str(self.signing_key)
        migration_config = {
            "server": {
                "env": [
                    {"name": "SERVER_UID", "value": server_uid},
                    {"name": "SERVER_PRIVATE_KEY", "value": server_private_key},
                ]
            }
        }
        return migration_config

    @classmethod
    def from_file(self, path: str | Path) -> Self:
        path = Path(path)
        if not path.exists():
            raise SyftException(f"File {str(path)} does not exist.")

        with open(path, "rb") as f:
            res: SyftObject = _deserialize(f.read(), from_bytes=True)

        if not isinstance(res, MigrationData):
            latest_version = SyftObjectRegistry.get_latest_version(  # type: ignore[unreachable]
                MigrationData.__canonical_name__
            )
            print("Upgrading MigrationData object to latest version...")
            res = res.migrate_to(latest_version)

        return res

    def save(self, path: str | Path, yaml_path: str | Path) -> SyftSuccess:
        if not self.includes_blobs:
            proceed = prompt_warning_message(
                "You are saving migration data without blob storage data. "
                "This means that any existing blobs will be missing when you load this data."
                "\nTo include blobs, call `download_blobs()` before saving.",
                confirm=True,
            )
            if not proceed:
                raise SyftException(message="Migration data not saved.")

        path = Path(path)
        with open(path, "wb") as f:
            f.write(_serialize(self, to_bytes=True))

        yaml_path = Path(yaml_path)
        migration_config = self.make_migration_config()
        with open(yaml_path, "w") as f:
            yaml.dump(migration_config, f)

        return SyftSuccess(message=f"Migration data saved to {str(path)}.")

    def download_blobs(self) -> None:
        for obj in self.blob_storage_objects:
            blob = self.download_blob(obj.id)
            self.blobs[obj.id] = blob
        return None

    def download_blob(self, obj_id: str) -> Any:
        api = self._get_api()
        blob_retrieval = api.services.blob_storage.read(obj_id)
        return blob_retrieval.read()

    def migrate_and_upload_blobs(self) -> SyftSuccess:
        for obj in self.blob_storage_objects:
            self.migrate_and_upload_blob(obj)
        return SyftSuccess(message="All blobs uploaded successfully.")

    def migrate_and_upload_blob(self, obj: BlobStorageEntry) -> SyftSuccess:
        api = self._get_api()

        if obj.id not in self.blobs:
            raise SyftException(
                public_message=f"Blob {obj.id} not found in migration data."
            )
        data = self.blobs[obj.id]

        migrated_obj = obj.migrate_to(BlobStorageEntry.__version__, Context())
        serialized = _serialize(data, to_bytes=True)
        size = sys.getsizeof(serialized)
        blob_create = CreateBlobStorageEntry.from_blob_storage_entry(migrated_obj)
        blob_create.file_size = size
        blob_deposit_object = api.services.blob_storage.allocate_for_user(
            blob_create, migrated_obj.uploaded_by
        )
        return blob_deposit_object.write(BytesIO(serialized)).unwrap()

    def get_items_by_canonical_name(self, canonical_name: str) -> list[SyftObject]:
        for k, v in self.store_objects.items():
            if k.__canonical_name__ == canonical_name:
                return v

        for k, v in self.action_objects.items():
            if k.__canonical_name__ == canonical_name:
                return v
        return []

    def get_metadata_by_canonical_name(self, canonical_name: str) -> StoreMetadata:
        for k, v in self.metadata.items():
            if k.__canonical_name__ == canonical_name:
                return v
        return StoreMetadata(
            object_type=SyftObject, permissions={}, storage_permissions={}
        )

    def copy_without_workerpools(self) -> "MigrationData":
        items_to_exclude = [
            WorkerPool.__canonical_name__,
            SyftWorkerImage.__canonical_name__,
            SyftWorker.__canonical_name__,
        ]

        store_objects = {
            k: v
            for k, v in self.store_objects.items()
            if k.__canonical_name__ not in items_to_exclude
        }
        metadata = {
            k: v
            for k, v in self.metadata.items()
            if k.__canonical_name__ not in items_to_exclude
        }
        return self.__class__(
            server_uid=self.server_uid,
            signing_key=self.signing_key,
            store_objects=store_objects,
            metadata=metadata,
            action_objects=self.action_objects,
            blob_storage_objects=self.blob_storage_objects,
            blobs=self.blobs,
        )

    def copy_without_blobs(self) -> "MigrationData":
        # Create a shallow copy of the MigrationData instance, removing blob-related data
        # This is required for sending the MigrationData to the backend.
        copy_data = self.__class__(
            server_uid=self.server_uid,
            signing_key=self.signing_key,
            store_objects=self.store_objects.copy(),
            metadata=self.metadata.copy(),
            action_objects=self.action_objects.copy(),
            blob_storage_objects=[],
            blobs={},
        )
        return copy_data


@serializable()
class MigrationDataV1(SyftObject):
    __canonical_name__ = "MigrationData"
    __version__ = SYFT_OBJECT_VERSION_1

    server_uid: UID
    signing_key: SyftSigningKey
    store_objects: dict[type[SyftObject], list[SyftObject]]
    metadata: dict[type[SyftObject], StoreMetadata]
    action_objects: dict[type[SyftObject], list[SyftObject]]
    blob_storage_objects: list[SyftObject]
    blobs: dict[UID, Any] = {}


@migrate(MigrationDataV1, MigrationData)
def migrate_migrationdata_v1_to_v2() -> list[Callable]:
    return [
        make_set_default("default_pool_name", DEFAULT_WORKER_POOL_NAME),
        make_set_default("syft_version", ""),
    ]
