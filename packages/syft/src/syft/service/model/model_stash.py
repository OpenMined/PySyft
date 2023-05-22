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
from .model import ModelCard
from .model import ModelUpdate

NamePartitionKey = PartitionKey(key="name", type_=str)
ActionIDsPartitionKey = PartitionKey(key="action_ids", type_=List[UID])


@instrument
@serializable()
class ModelCardStash(BaseUIDStoreStash):
    object_type = ModelCard
    settings: PartitionSettings = PartitionSettings(
        name=ModelCard.__canonical_name__, object_type=ModelCard
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_name(
        self, credentials: SyftVerifyKey, name: str
    ) -> Result[Optional[ModelCard], str]:
        qks = QueryKeys(qks=[NamePartitionKey.with_obj(name)])
        return self.query_one(credentials=credentials, qks=qks)

    def update(
        self, credentials: SyftVerifyKey, model_update: ModelUpdate
    ) -> Result[ModelCard, str]:
        res = self.check_type(model_update, ModelUpdate)
        # we dont use and_then logic here as it is hard because of the order of the arguments
        if res.is_err():
            return res
        return super().update(credentials=credentials, obj=res.ok())

    def search_action_ids(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[List[ModelCard], str]:
        qks = QueryKeys(qks=[ActionIDsPartitionKey.with_obj(uid)])
        return self.query_all(credentials=credentials, qks=qks)
