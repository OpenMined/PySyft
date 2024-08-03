# stdlib

# third party
from result import Result

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.document_store import (
    BaseUIDStoreStash,
    DocumentStore,
    PartitionKey,
    PartitionSettings,
    QueryKeys,
)
from ...types.uid import UID
from ..action.action_permissions import ActionObjectPermission, ActionPermission
from .worker_pool import WorkerPool

PoolNamePartitionKey = PartitionKey(key="name", type_=str)
PoolImageIDPartitionKey = PartitionKey(key="image_id", type_=UID)


@serializable(canonical_name="SyftWorkerPoolStash", version=1)
class SyftWorkerPoolStash(BaseUIDStoreStash):
    object_type = WorkerPool
    settings: PartitionSettings = PartitionSettings(
        name=WorkerPool.__canonical_name__,
        object_type=WorkerPool,
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_name(
        self, credentials: SyftVerifyKey, pool_name: str,
    ) -> Result[WorkerPool | None, str]:
        qks = QueryKeys(qks=[PoolNamePartitionKey.with_obj(pool_name)])
        return self.query_one(credentials=credentials, qks=qks)

    def set(
        self,
        credentials: SyftVerifyKey,
        obj: WorkerPool,
        add_permissions: list[ActionObjectPermission] | None = None,
        add_storage_permission: bool = True,
        ignore_duplicates: bool = False,
    ) -> Result[WorkerPool, str]:
        # By default all worker pools have all read permission
        add_permissions = [] if add_permissions is None else add_permissions
        add_permissions.append(
            ActionObjectPermission(uid=obj.id, permission=ActionPermission.ALL_READ),
        )
        return super().set(
            credentials,
            obj,
            add_permissions=add_permissions,
            add_storage_permission=add_storage_permission,
            ignore_duplicates=ignore_duplicates,
        )

    def get_by_image_uid(
        self, credentials: SyftVerifyKey, image_uid: UID,
    ) -> list[WorkerPool]:
        qks = QueryKeys(qks=[PoolImageIDPartitionKey.with_obj(image_uid)])
        return self.query_all(credentials=credentials, qks=qks)
