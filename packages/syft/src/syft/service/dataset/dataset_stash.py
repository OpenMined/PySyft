# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.document_store import DocumentStore
from ...store.document_store import NewBaseUIDStoreStash
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.result import as_result
from ...types.uid import UID
from .dataset import Dataset
from .dataset import DatasetUpdate

NamePartitionKey = PartitionKey(key="name", type_=str)
ActionIDsPartitionKey = PartitionKey(key="action_ids", type_=list[UID])


@serializable(canonical_name="DatasetStash", version=1)
class DatasetStash(NewBaseUIDStoreStash):
    object_type = Dataset
    settings: PartitionSettings = PartitionSettings(
        name=Dataset.__canonical_name__, object_type=Dataset
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    @as_result(StashException, NotFoundException)
    def get_by_name(self, credentials: SyftVerifyKey, name: str) -> Dataset:
        qks = QueryKeys(qks=[NamePartitionKey.with_obj(name)])
        return self.query_one(credentials=credentials, qks=qks).unwrap()

    @as_result(StashException)
    def search_action_ids(self, credentials: SyftVerifyKey, uid: UID) -> list[Dataset]:
        qks = QueryKeys(qks=[ActionIDsPartitionKey.with_obj(uid)])
        return self.query_all(credentials=credentials, qks=qks).unwrap()

    @as_result(StashException)
    def get_all(
        self,
        credentials: SyftVerifyKey,
        order_by: PartitionKey | None = None,
        has_permission: bool = False,
    ) -> list:
        result = super().get_all(credentials, order_by, has_permission).unwrap()
        filtered_datasets = [dataset for dataset in result if not dataset.to_be_deleted]
        return filtered_datasets

    # FIX: This shouldn't be the update method, it just marks the dataset for deletion
    @as_result(StashException)
    def update(
        self,
        credentials: SyftVerifyKey,
        obj: DatasetUpdate,
        has_permission: bool = False,
    ) -> Dataset:
        _obj = self.check_type(obj, DatasetUpdate).unwrap()
        # FIX: This method needs a revamp
        qk = self.partition.store_query_key(obj)
        return self.partition.update(
            credentials=credentials, qk=qk, obj=_obj, has_permission=has_permission
        ).unwrap()
