# stdlib
from enum import Enum
from typing import Any

# third party
from result import Ok
from result import Result

# relative
from ...node.credentials import SyftVerifyKey
from ...node.worker_settings import WorkerSettings
from ...serde.serializable import serializable
from ...store.document_store import BaseStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...store.document_store import UIDPartitionKey
from ...store.linked_obj import LinkedObject
from ...types.syft_object import SYFT_OBJECT_VERSION_3
from ...types.syft_object import SYFT_OBJECT_VERSION_4
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_permissions import ActionObjectPermission
from ..response import SyftError
from ..response import SyftSuccess


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


@instrument
@serializable()
class QueueStash(BaseStash):
    object_type = QueueItem
    settings: PartitionSettings = PartitionSettings(
        name=QueueItem.__canonical_name__, object_type=QueueItem
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def set_result(
        self,
        credentials: SyftVerifyKey,
        item: QueueItem,
        add_permissions: list[ActionObjectPermission] | None = None,
    ) -> Result[QueueItem | None, str]:
        if item.resolved:
            valid = self.check_type(item, self.object_type)
            if valid.is_err():
                return SyftError(message=valid.err())
            return super().update(credentials, item, add_permissions)
        return None

    def set_placeholder(
        self,
        credentials: SyftVerifyKey,
        item: QueueItem,
        add_permissions: list[ActionObjectPermission] | None = None,
    ) -> Result[QueueItem, str]:
        # 🟡 TODO 36: Needs distributed lock
        if not item.resolved:
            exists = self.get_by_uid(credentials, item.id)
            if exists.is_ok() and exists.ok() is None:
                valid = self.check_type(item, self.object_type)
                if valid.is_err():
                    return SyftError(message=valid.err())
                return super().set(credentials, item, add_permissions)
        return item

    def get_by_uid(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[QueueItem | None, str]:
        qks = QueryKeys(qks=[UIDPartitionKey.with_obj(uid)])
        item = self.query_one(credentials=credentials, qks=qks)
        return item

    def pop(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[QueueItem | None, str]:
        item = self.get_by_uid(credentials=credentials, uid=uid)
        self.delete_by_uid(credentials=credentials, uid=uid)
        return item

    def pop_on_complete(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[QueueItem | None, str]:
        item = self.get_by_uid(credentials=credentials, uid=uid)
        if item.is_ok():
            queue_item = item.ok()
            if queue_item.status == Status.COMPLETED:
                self.delete_by_uid(credentials=credentials, uid=uid)
        return item

    def delete_by_uid(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[SyftSuccess, str]:
        qk = UIDPartitionKey.with_obj(uid)
        result = super().delete(credentials=credentials, qk=qk)
        if result.is_ok():
            return Ok(SyftSuccess(message=f"ID: {uid} deleted"))
        return result

    def get_by_status(
        self, credentials: SyftVerifyKey, status: Status
    ) -> Result[list[QueueItem], str]:
        qks = QueryKeys(qks=StatusPartitionKey.with_obj(status))

        return self.query_all(credentials=credentials, qks=qks)
