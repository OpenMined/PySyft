# stdlib
from enum import Enum
from typing import Any

# relative
from ...node.credentials import SyftVerifyKey
from ...node.worker_settings import WorkerSettings
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...store.document_store import NewBaseStash
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...store.document_store import UIDPartitionKey
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...store.linked_obj import LinkedObject
from ...types.result import as_result
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_3
from ...types.syft_object import SYFT_OBJECT_VERSION_4
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_permissions import ActionObjectPermission
from ..response import SyftError
from ..response import SyftException


@serializable()
class Status(str, Enum):
    CREATED = "created"
    PROCESSING = "processing"
    ERRORED = "errored"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"


StatusPartitionKey = PartitionKey(key="status", type_=Status)


@serializable()
class QueueItem(SyftObject):
    __canonical_name__ = "QueueItem"
    __version__ = SYFT_OBJECT_VERSION_4

    __attr_searchable__ = ["status"]

    id: UID
    node_uid: UID
    result: Any | None = None
    resolved: bool = False
    status: Status = Status.CREATED

    method: str
    service: str
    args: list
    kwargs: dict[str, Any]
    job_id: UID | None = None
    worker_settings: WorkerSettings | None = None
    has_execute_permissions: bool = False
    worker_pool: LinkedObject

    def __repr__(self) -> str:
        return f"<QueueItem: {self.id}>: {self.status}"

    def _repr_markdown_(self, wrap_as_python: bool = True, indent: int = 0) -> str:
        return f"<QueueItem: {self.id}>: {self.status}"

    @property
    def is_action(self) -> bool:
        return self.service_path == "Action" and self.method_name == "execute"

    @property
    def action(self) -> Any | SyftError:
        if self.is_action:
            return self.kwargs["action"]
        return SyftError(message="QueueItem not an Action")


@serializable()
class ActionQueueItem(QueueItem):
    __canonical_name__ = "ActionQueueItem"
    __version__ = SYFT_OBJECT_VERSION_3

    method: str = "execute"
    service: str = "actionservice"


@serializable()
class APIEndpointQueueItem(QueueItem):
    __canonical_name__ = "APIEndpointQueueItem"
    __version__ = SYFT_OBJECT_VERSION_1

    method: str
    service: str = "apiservice"


@instrument
@serializable()
class QueueStash(NewBaseStash):
    object_type = QueueItem
    settings: PartitionSettings = PartitionSettings(
        name=QueueItem.__canonical_name__, object_type=QueueItem
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    # FIX: Check the return value for None. set_result is used extensively
    @as_result(StashException)
    def set_result(
        self,
        credentials: SyftVerifyKey,
        item: QueueItem,
        add_permissions: list[ActionObjectPermission] | None = None,
    ) -> QueueItem | None:
        if item.resolved:
            self.check_type(item, self.object_type).unwrap()
            return super().update(credentials, item, add_permissions).unwrap()
        # TODO: should we log this?
        return None

    @as_result(SyftException)
    def set_placeholder(
        self,
        credentials: SyftVerifyKey,
        item: QueueItem,
        add_permissions: list[ActionObjectPermission] | None = None,
    ) -> QueueItem:
        # 🟡 TODO 36: Needs distributed lock
        if not item.resolved:
            try:
                self.get_by_uid(credentials, item.id).unwrap()
            except NotFoundException:
                self.check_type(item, self.object_type)
                return super().set(credentials, item, add_permissions).unwrap()
        return item

    @as_result(StashException)
    def get_by_uid(self, credentials: SyftVerifyKey, uid: UID) -> QueueItem:
        qks = QueryKeys(qks=[UIDPartitionKey.with_obj(uid)])
        return self.query_one(credentials=credentials, qks=qks).unwrap()

    as_result(StashException)

    @as_result(StashException)
    def pop(self, credentials: SyftVerifyKey, uid: UID) -> QueueItem | None:
        try:
            item = self.get_by_uid(credentials=credentials, uid=uid).unwrap()
        except NotFoundException:
            # TODO: Handle NotfoundException in code?
            return None

        self.delete_by_uid(credentials=credentials, uid=uid).unwrap()
        return item

    @as_result(StashException)
    def pop_on_complete(self, credentials: SyftVerifyKey, uid: UID) -> QueueItem | None:
        queue_item = self.get_by_uid(credentials=credentials, uid=uid).unwrap()
        if queue_item.status == Status.COMPLETED:
            self.delete_by_uid(credentials=credentials, uid=uid)
        return queue_item

    @as_result(StashException)
    def delete_by_uid(self, credentials: SyftVerifyKey, uid: UID) -> UID:
        qk = UIDPartitionKey.with_obj(uid)
        return super().delete(credentials=credentials, qk=qk).unwrap()

    @as_result(StashException)
    def get_by_status(
        self, credentials: SyftVerifyKey, status: Status
    ) -> list[QueueItem]:
        qks = QueryKeys(qks=StatusPartitionKey.with_obj(status))
        return self.query_all(credentials=credentials, qks=qks).unwrap()
