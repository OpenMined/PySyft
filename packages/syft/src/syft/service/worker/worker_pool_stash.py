# stdlib
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
from .worker_pool import WorkerPool

PoolNamePartitionKey = PartitionKey(key="name", type_=str)


@serializable()
class SyftWorkerPoolStash(BaseUIDStoreStash):
    object_type = WorkerPool
    settings: PartitionSettings = PartitionSettings(
        name=WorkerPool.__canonical_name__,
        object_type=WorkerPool,
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_name(
        self, credentials: SyftVerifyKey, pool_name: str
    ) -> Result[Optional[WorkerPool], str]:
        qks = QueryKeys(qks=[PoolNamePartitionKey.with_obj(pool_name)])
        return self.query_one(credentials=credentials, qks=qks)
