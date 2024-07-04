# stdlib

# third party
from result import Result

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash, NewBaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_permissions import ActionObjectPermission
from .settings import NodeSettings

NamePartitionKey = PartitionKey(key="name", type_=str)
ActionIDsPartitionKey = PartitionKey(key="action_ids", type_=list[UID])


@instrument
@serializable()
class SettingsStash(NewBaseUIDStoreStash):
    object_type = NodeSettings
    settings: PartitionSettings = PartitionSettings(
        name=NodeSettings.__canonical_name__, object_type=NodeSettings
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def update(
        self,
        credentials: SyftVerifyKey,
        settings: NodeSettings,
        has_permission: bool = False,
    ) -> NodeSettings:
        obj = self.check_type(settings, self.object_type).unwrap()
        return super().update(credentials=credentials, obj=obj).unwrap()
