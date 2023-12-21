# stdlib
from typing import List
from typing import Optional
from typing import Union

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
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from .worker_pool import SyftWorker

WorkerContainerNamePartitionKey = PartitionKey(key="container_name", type_=str)


@instrument
@serializable()
class WorkerStash(BaseUIDStoreStash):
    object_type = SyftWorker
    settings: PartitionSettings = PartitionSettings(
        name=SyftWorker.__canonical_name__, object_type=SyftWorker
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def set(
        self,
        credentials: SyftVerifyKey,
        obj: SyftWorker,
        add_permissions: Union[List[ActionObjectPermission], None] = None,
        ignore_duplicates: bool = False,
    ) -> Result[SyftWorker, str]:
        # By default all worker pools have all read permission
        add_permissions = [] if add_permissions is None else add_permissions
        add_permissions.append(
            ActionObjectPermission(uid=obj.id, permission=ActionPermission.ALL_READ)
        )
        return super().set(credentials, obj, add_permissions, ignore_duplicates)

    def get_worker_by_name(
        self, credentials: SyftVerifyKey, worker_name: str
    ) -> Result[Optional[SyftWorker], str]:
        qks = QueryKeys(qks=[WorkerContainerNamePartitionKey.with_obj(worker_name)])
        return self.query_one(credentials=credentials, qks=qks)
