# stdlib

# third party
from result import Result

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SyftMigrationRegistry
from ...types.syft_object import SyftObject
from ..action.action_permissions import ActionObjectPermission


@serializable()
class SyftObjectMigrationState(SyftObject):
    __canonical_name__ = "SyftObjectMigrationState"
    __version__ = SYFT_OBJECT_VERSION_2

    __attr_unique__ = ["canonical_name"]

    canonical_name: str
    current_version: int

    @property
    def latest_version(self) -> int | None:
        available_versions = SyftMigrationRegistry.get_versions(
            canonical_name=self.canonical_name,
        )
        if not available_versions:
            return None

        return sorted(available_versions, reverse=True)[0]

    @property
    def supported_versions(self) -> list:
        return SyftMigrationRegistry.get_versions(self.canonical_name)


KlassNamePartitionKey = PartitionKey(key="canonical_name", type_=str)


@serializable()
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
