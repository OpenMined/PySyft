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
from ...util.telemetry import instrument
from .worker import DockerWorker

WorkerContainerNamePartitionKey = PartitionKey(key="container_name", type_=str)


@instrument
@serializable()
class WorkerStash(BaseUIDStoreStash):
    object_type = DockerWorker
    settings: PartitionSettings = PartitionSettings(
        name=DockerWorker.__canonical_name__, object_type=DockerWorker
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_worker_id_by_name(
        self, credentials: SyftVerifyKey, worker_name: str
    ) -> Result[Optional[DockerWorker], str]:
        qks = QueryKeys(qks=[WorkerContainerNamePartitionKey.with_obj(worker_name)])
        return self.query_one(credentials=credentials, qks=qks)
