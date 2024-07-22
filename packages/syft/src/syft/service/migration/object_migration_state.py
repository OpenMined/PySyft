# stdlib
from io import BytesIO
from pathlib import Path
import sys
from typing import Any

# third party
from result import Result
from typing_extensions import Self
import yaml

# relative
from ...serde.deserialize import _deserialize
from ...serde.serializable import serializable
from ...serde.serialize import _serialize
from ...server.credentials import SyftSigningKey
from ...server.credentials import SyftVerifyKey
from ...store.document_store import BaseStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...types.blob_storage import BlobStorageEntry
from ...types.blob_storage import CreateBlobStorageEntry
from ...types.syft_object import Context
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftBaseObject
from ...types.syft_object import SyftObject
from ...types.syft_object_registry import SyftObjectRegistry
from ...types.uid import UID
from ...util.util import prompt_warning_message
from ..action.action_permissions import ActionObjectPermission
from ..response import SyftError
from ..response import SyftSuccess


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


KlassNamePartitionKey = PartitionKey(key="canonical_name", type_=str)


@serializable(canonical_name="SyftMigrationStateStash", version=1)
class SyftMigrationStateStash(BaseStash):
    object_type = SyftObjectMigrationState
    settings: PartitionSettings = PartitionSettings(
        name=SyftObjectMigrationState.__canonical_name__,
        object_type=SyftObjectMigrationState,
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def set(
        self,
        credentials: SyftVerifyKey,
        migration_state: SyftObjectMigrationState,
        add_permissions: list[ActionObjectPermission] | None = None,
        add_storage_permission: bool = True,
        ignore_duplicates: bool = False,
    ) -> Result[SyftObjectMigrationState, str]:
        res = self.check_type(migration_state, self.object_type)
        # we dont use and_then logic here as it is hard because of the order of the arguments
        if res.is_err():
            return res
        return super().set(
            credentials=credentials,
            obj=res.ok(),
            add_permissions=add_permissions,
            add_storage_permission=add_storage_permission,
            ignore_duplicates=ignore_duplicates,
        )

    def get_by_name(
        self, canonical_name: str, credentials: SyftVerifyKey
    ) -> Result[SyftObjectMigrationState, str]:
        qks = KlassNamePartitionKey.with_obj(canonical_name)
        return self.query_one(credentials=credentials, qks=qks)


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
    __version__ = SYFT_OBJECT_VERSION_1

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
    def from_file(self, path: str | Path) -> Self | SyftError:
        path = Path(path)
        if not path.exists():
            return SyftError(f"File {str(path)} does not exist.")

        with open(path, "rb") as f:
            res: MigrationData = _deserialize(f.read(), from_bytes=True)

        return res

    def save(self, path: str | Path, yaml_path: str | Path) -> SyftSuccess | SyftError:
        if not self.includes_blobs:
            proceed = prompt_warning_message(
                "You are saving migration data without blob storage data. "
                "This means that any existing blobs will be missing when you load this data."
                "\nTo include blobs, call `download_blobs()` before saving.",
                confirm=True,
            )
            if not proceed:
                return SyftError(message="Migration data not saved.")

        path = Path(path)
        with open(path, "wb") as f:
            f.write(_serialize(self, to_bytes=True))

        yaml_path = Path(yaml_path)
        migration_config = self.make_migration_config()
        with open(yaml_path, "w") as f:
            yaml.dump(migration_config, f)

        return SyftSuccess(message=f"Migration data saved to {str(path)}.")

    def download_blobs(self) -> None | SyftError:
        for obj in self.blob_storage_objects:
            blob = self.download_blob(obj.id)
            if isinstance(blob, SyftError):
                return blob
            self.blobs[obj.id] = blob
        return None

    def download_blob(self, obj_id: str) -> Any | SyftError:
        api = self._get_api()
        if isinstance(api, SyftError):
            return api

        blob_retrieval = api.services.blob_storage.read(obj_id)
        if isinstance(blob_retrieval, SyftError):
            return blob_retrieval
        return blob_retrieval.read()

    def migrate_and_upload_blobs(self) -> SyftSuccess | SyftError:
        for obj in self.blob_storage_objects:
            upload_result = self.migrate_and_upload_blob(obj)
            if isinstance(upload_result, SyftError):
                return upload_result
        return SyftSuccess(message="All blobs uploaded successfully.")

    def migrate_and_upload_blob(self, obj: BlobStorageEntry) -> SyftSuccess | SyftError:
        api = self._get_api()
        if isinstance(api, SyftError):
            return api

        if obj.id not in self.blobs:
            return SyftError(f"Blob {obj.id} not found in migration data.")
        data = self.blobs[obj.id]

        migrated_obj = obj.migrate_to(BlobStorageEntry.__version__, Context())
        serialized = _serialize(data, to_bytes=True)
        size = sys.getsizeof(serialized)
        blob_create = CreateBlobStorageEntry.from_blob_storage_entry(migrated_obj)
        blob_create.file_size = size
        blob_deposit_object = api.services.blob_storage.allocate_for_user(
            blob_create, migrated_obj.uploaded_by
        )

        if isinstance(blob_deposit_object, SyftError):
            return blob_deposit_object
        return blob_deposit_object.write(BytesIO(serialized))

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
