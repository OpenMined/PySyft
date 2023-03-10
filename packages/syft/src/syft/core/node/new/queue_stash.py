# stdlib
from typing import Any
from typing import Optional
from typing import Union

# third party
from result import Ok
from result import Result

# relative
from ....telemetry import instrument
from .api import APIRegistry
from .api import SyftAPICall
from .document_store import BaseStash
from .document_store import DocumentStore
from .document_store import PartitionSettings
from .document_store import QueryKeys
from .document_store import UIDPartitionKey
from .response import SyftNotReady
from .response import SyftSuccess
from .serializable import serializable
from .syft_object import SYFT_OBJECT_VERSION_1
from .syft_object import SyftObject
from .uid import UID


@serializable(recursive_serde=True)
class QueueItem(SyftObject):
    __canonical_name__ = "QueueItem"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_state__ = ["id", "node_uid", "result", "resolved"]

    id: UID
    node_uid: UID
    result: Optional[Any]
    resolved: bool = False

    def fetch(self) -> None:
        api = APIRegistry.api_for(node_uid=self.node_uid)
        call = SyftAPICall(
            node_uid=self.node_uid,
            path="queue",
            args=[],
            kwargs={"uid": self.id},
            blocking=True,
        )
        result = api.make_call(call)
        if isinstance(result, QueueItem) and result.resolved:
            self.resolved = True
            self.result = result.result

    @property
    def resolve(self) -> Union[Any, SyftNotReady]:
        if not self.resolved:
            self.fetch()

        if self.resolved:
            return self.result
        return SyftNotReady(message=f"{self.id} not ready yet.")


@instrument
@serializable(recursive_serde=True)
class QueueStash(BaseStash):
    object_type = QueueItem
    settings: PartitionSettings = PartitionSettings(
        name=QueueItem.__canonical_name__, object_type=QueueItem
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def set_result(self, item: QueueItem) -> Result[Optional[QueueItem], str]:
        if item.resolved:
            return self.check_type(item, self.object_type).and_then(super().set)
        return None

    # def set_placeholder(self, item: QueueItem) -> Result[QueueItem, str]:
    #     # 🟡 TODO 36: Needs distributed lock
    #     if not item.resolved:
    #         exists = self.get_by_uid(item.id)
    #         if exists.is_ok() and exists.ok() is None:
    #             return self.check_type(item, self.object_type).and_then(super().set)
    #     return item

    def get_by_uid(self, uid: UID) -> Result[Optional[QueueItem], str]:
        qks = QueryKeys(qks=[UIDPartitionKey.with_obj(uid)])
        item = self.query_one(qks=qks)
        return item

    def pop(self, uid: UID) -> Result[Optional[QueueItem], str]:
        item = self.get_by_uid(uid)
        self.delete_by_uid(uid)
        return item

    def delete_by_uid(self, uid: UID) -> Result[SyftSuccess, str]:
        qk = UIDPartitionKey.with_obj(uid)
        result = super().delete(qk=qk)
        if result.is_ok():
            return Ok(SyftSuccess(message=f"ID: {uid} deleted"))
        return result
