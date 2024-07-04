# stdlib

# third party
from pytest import Stash
from result import Result
from syft.store.document_store_errors import NotFoundException, StashException
from syft.types.errors import SyftException
from syft.types.result import as_result

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash, NewBaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...types.uid import UID
from ...util.telemetry import instrument
from .dataset import Dataset
from .dataset import DatasetUpdate

NamePartitionKey = PartitionKey(key="name", type_=str)
ActionIDsPartitionKey = PartitionKey(key="action_ids", type_=list[UID])


@instrument
@serializable()
class DatasetStash(NewBaseUIDStoreStash):
    object_type = Dataset
    settings: PartitionSettings = PartitionSettings(
        name=Dataset.__canonical_name__, object_type=Dataset
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    @as_result(StashException, NotFoundException)
    def get_by_name(
        self, credentials: SyftVerifyKey, name: str
    ) -> Dataset:
        qks = QueryKeys(qks=[NamePartitionKey.with_obj(name)])
        return self.query_one(credentials=credentials, qks=qks).unwrap()

    @as_result(StashException, NotFoundException)
    def update(
        self,
        credentials: SyftVerifyKey,
        dataset_update: DatasetUpdate,
        has_permission: bool = False,
    ) -> Dataset:
        res = self.check_type(dataset_update, DatasetUpdate).unwrap()
        return super().update(credentials=credentials, obj=res).unwrap()

    @as_result(StashException)
    def search_action_ids(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> list[Dataset]:
        qks = QueryKeys(qks=[ActionIDsPartitionKey.with_obj(uid)])
        return self.query_all(credentials=credentials, qks=qks).unwrap()
