# stdlib
from typing import List
from typing import Optional

# third party
from result import Result

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...types.uid import UID
from ...util.telemetry import instrument
from .event import Event


@instrument
@serializable()
class EventStash(BaseUIDStoreStash):
    object_type = Event
    settings: PartitionSettings = PartitionSettings(
        name=Event.__canonical_name__, object_type=Event
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)