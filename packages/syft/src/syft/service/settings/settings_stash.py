# stdlib

# third party

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...types.uid import UID
from ...util.telemetry import instrument
from ..job.base_stash import ObjectStash
from .settings import ServerSettings

NamePartitionKey = PartitionKey(key="name", type_=str)
ActionIDsPartitionKey = PartitionKey(key="action_ids", type_=list[UID])


@instrument
@serializable(canonical_name="SettingsStashSQL", version=1)
class SettingsStash(ObjectStash[ServerSettings]):
    object_type = ServerSettings
    settings: PartitionSettings = PartitionSettings(
        name=ServerSettings.__canonical_name__, object_type=ServerSettings
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)
