# stdlib

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
from ...util.telemetry import instrument
from .model import Model

NamePartitionKey = PartitionKey(key="name", type_=str)


@instrument
@serializable()
class ModelStash(BaseUIDStoreStash):
    object_type = Model
    settings: PartitionSettings = PartitionSettings(
        name=Model.__canonical_name__, object_type=Model
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_name(
        self, credentials: SyftVerifyKey, name: str
    ) -> Result[Model | None, str]:
        qks = QueryKeys(qks=[NamePartitionKey.with_obj(name)])
        return self.query_one(credentials=credentials, qks=qks)
