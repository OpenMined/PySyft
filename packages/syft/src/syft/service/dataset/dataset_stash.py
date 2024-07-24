# stdlib

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.document_store import BaseUIDStoreStash
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
@serializable(canonical_name="DatasetStash", version=1)
class DatasetStash(BaseUIDStoreStash):
    object_type = Dataset
    settings: PartitionSettings = PartitionSettings(
        name=Dataset.__canonical_name__, object_type=Dataset
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_name(
        self, credentials: SyftVerifyKey, name: str
    ) -> Result[Dataset | None, str]:
        qks = QueryKeys(qks=[NamePartitionKey.with_obj(name)])
        return self.query_one(credentials=credentials, qks=qks)

    def update(
        self,
        credentials: SyftVerifyKey,
        dataset_update: DatasetUpdate,
        has_permission: bool = False,
    ) -> Result[Dataset, str]:
        res = self.check_type(dataset_update, DatasetUpdate)
        # we dont use and_then logic here as it is hard because of the order of the arguments
        if res.is_err():
            return res
        return super().update(credentials=credentials, obj=res.ok())

    def search_action_ids(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[list[Dataset], str]:
        qks = QueryKeys(qks=[ActionIDsPartitionKey.with_obj(uid)])
        return self.query_all(credentials=credentials, qks=qks)

    def get_all(
        self,
        credentials: SyftVerifyKey,
        order_by: PartitionKey | None = None,
        has_permission: bool = False,
    ) -> Ok[list] | Err[str]:
        result = super().get_all(credentials, order_by, has_permission)

        if result.is_err():
            return result
        filtered_datasets = [
            dataset for dataset in result.ok_value if not dataset.to_be_deleted
        ]
        return Ok(filtered_datasets)
