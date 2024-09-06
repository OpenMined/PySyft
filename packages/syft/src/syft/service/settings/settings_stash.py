# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.document_store import DocumentStore
from ...store.document_store import NewBaseUIDStoreStash
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store_errors import StashException
from ...types.result import as_result
from ...types.uid import UID
from .settings import ServerSettings

NamePartitionKey = PartitionKey(key="name", type_=str)
ActionIDsPartitionKey = PartitionKey(key="action_ids", type_=list[UID])


@serializable(canonical_name="SettingsStash", version=1)
class SettingsStash(NewBaseUIDStoreStash):
    object_type = ServerSettings
    settings: PartitionSettings = PartitionSettings(
        name=ServerSettings.__canonical_name__, object_type=ServerSettings
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    # Should we have this at all?
    @as_result(StashException)
    def update(
        self,
        credentials: SyftVerifyKey,
        settings: ServerSettings,
        has_permission: bool = False,
    ) -> ServerSettings:
        obj = self.check_type(settings, self.object_type).unwrap()
        return super().update(credentials=credentials, obj=obj).unwrap()
