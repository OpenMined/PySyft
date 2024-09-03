# stdlib

# third party

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
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from .worker_pool import ConsumerState
from .worker_pool import SyftWorker

WorkerContainerNamePartitionKey = PartitionKey(key="container_name", type_=str)


@serializable(canonical_name="WorkerStash", version=1)
class WorkerStash(NewBaseUIDStoreStash):
    object_type = SyftWorker
    settings: PartitionSettings = PartitionSettings(
        name=SyftWorker.__canonical_name__, object_type=SyftWorker
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    @as_result(StashException)
    def set(
        self,
        credentials: SyftVerifyKey,
        obj: SyftWorker,
        add_permissions: list[ActionObjectPermission] | None = None,
        add_storage_permission: bool = True,
        ignore_duplicates: bool = False,
    ) -> SyftWorker:
        # By default all worker pools have all read permission
        add_permissions = [] if add_permissions is None else add_permissions
        add_permissions.append(
            ActionObjectPermission(uid=obj.id, permission=ActionPermission.ALL_READ)
        )
        return (
            super()
            .set(
                credentials,
                obj,
                add_permissions=add_permissions,
                ignore_duplicates=ignore_duplicates,
                add_storage_permission=add_storage_permission,
            )
            .unwrap()
        )

    @as_result(StashException, NotFoundException)
    def get_worker_by_name(
        self, credentials: SyftVerifyKey, worker_name: str
    ) -> SyftWorker:
        qks = QueryKeys(qks=[WorkerContainerNamePartitionKey.with_obj(worker_name)])
        return self.query_one(credentials=credentials, qks=qks).unwrap()

    @as_result(StashException, NotFoundException)
    def update_consumer_state(
        self, credentials: SyftVerifyKey, worker_uid: UID, consumer_state: ConsumerState
    ) -> SyftWorker:
        worker = self.get_by_uid(credentials=credentials, uid=worker_uid).unwrap()
        worker.consumer_state = consumer_state
        return self.update(credentials=credentials, obj=worker).unwrap()
