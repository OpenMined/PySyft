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
from ...types.uid import UID
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from .worker_pool import WorkerPool

PoolNamePartitionKey = PartitionKey(key="name", type_=str)
PoolImageIDPartitionKey = PartitionKey(key="image_id", type_=UID)


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

    def set(
        self,
        credentials: SyftVerifyKey,
        obj: WorkerPool,
        add_permissions: Union[List[ActionObjectPermission], None] = None,
        ignore_duplicates: bool = False,
    ) -> Result[WorkerPool, str]:
        # By default all worker pools have all read permission
        add_permissions = [] if add_permissions is None else add_permissions
        add_permissions.append(
            ActionObjectPermission(uid=obj.id, permission=ActionPermission.ALL_READ)
        )
        return super().set(credentials, obj, add_permissions, ignore_duplicates)

    def get_by_image_uid(
        self, credentials: SyftVerifyKey, image_uid: UID
    ) -> List[WorkerPool]:
        qks = QueryKeys(qks=[PoolImageIDPartitionKey.with_obj(image_uid)])
        return self.query_all(credentials=credentials, qks=qks)
