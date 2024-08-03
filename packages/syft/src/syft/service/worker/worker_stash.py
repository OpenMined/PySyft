# stdlib

# third party
from result import Err, Ok, Result

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
from ...util.telemetry import instrument
from ..action.action_permissions import ActionObjectPermission, ActionPermission
from .worker_pool import ConsumerState, SyftWorker

WorkerContainerNamePartitionKey = PartitionKey(key="container_name", type_=str)


@instrument
@serializable(canonical_name="WorkerStash", version=1)
class WorkerStash(BaseUIDStoreStash):
    object_type = SyftWorker
    settings: PartitionSettings = PartitionSettings(
        name=SyftWorker.__canonical_name__, object_type=SyftWorker,
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def set(
        self,
        credentials: SyftVerifyKey,
        obj: SyftWorker,
        add_permissions: list[ActionObjectPermission] | None = None,
        add_storage_permission: bool = True,
        ignore_duplicates: bool = False,
    ) -> Result[SyftWorker, str]:
        # By default all worker pools have all read permission
        add_permissions = [] if add_permissions is None else add_permissions
        add_permissions.append(
            ActionObjectPermission(uid=obj.id, permission=ActionPermission.ALL_READ),
        )
        return super().set(
            credentials,
            obj,
            add_permissions=add_permissions,
            ignore_duplicates=ignore_duplicates,
            add_storage_permission=add_storage_permission,
        )

    def get_worker_by_name(
        self, credentials: SyftVerifyKey, worker_name: str,
    ) -> Result[SyftWorker | None, str]:
        qks = QueryKeys(qks=[WorkerContainerNamePartitionKey.with_obj(worker_name)])
        return self.query_one(credentials=credentials, qks=qks)

    def update_consumer_state(
        self, credentials: SyftVerifyKey, worker_uid: UID, consumer_state: ConsumerState,
    ) -> Result[str, str]:
        res = self.get_by_uid(credentials=credentials, uid=worker_uid)
        if res.is_err():
            return Err(
                f"Failed to retrieve Worker with id: {worker_uid}. Error: {res.err()}",
            )
        worker: SyftWorker | None = res.ok()
        if worker is None:
            return Err(f"Worker with id: {worker_uid} not found")
        worker.consumer_state = consumer_state
        update_res = self.update(credentials=credentials, obj=worker)
        if update_res.is_err():
            return Err(
                f"Failed to update Worker with id: {worker_uid}. Error: {update_res.err()}",
            )
        return Ok(f"Successfully updated Worker with id: {worker_uid}")
