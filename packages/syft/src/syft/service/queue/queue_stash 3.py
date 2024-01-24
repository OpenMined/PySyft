# stdlib
from enum import Enum
from typing import Any
from typing import List
from typing import Optional
from typing import Union

# third party
from result import Ok
from result import Result

# relative
from ...client.api import APIRegistry
from ...client.api import SyftAPICall
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...store.document_store import UIDPartitionKey
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_permissions import ActionObjectPermission
from ..response import SyftError
from ..response import SyftNotReady
from ..response import SyftSuccess


@serializable()
class Status(str, Enum):
    CREATED = "created"
    PROCESSING = "processing"
    ERRORED = "errored"
    COMPLETED = "completed"


@serializable()
class QueueItem(SyftObject):
    __canonical_name__ = "QueueItem"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    node_uid: UID
    result: Optional[Any]
    resolved: bool = False
    status: Status = Status.CREATED

    def fetch(self) -> None:
        api = APIRegistry.api_for(
            node_uid=self.node_uid,
            user_verify_key=self.syft_client_verify_key,
        )
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
            self.status = result.status
        return result

    @property
    def resolve(self) -> Union[Any, SyftNotReady]:
        if not self.resolved:
            self.fetch()

        if self.resolved:
            return self.result.message
        return SyftNotReady(message=f"{self.id} not ready yet.")


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
        add_permissions: Optional[List[ActionObjectPermission]] = None,
    ) -> Result[Optional[QueueItem], str]:
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
        add_permissions: Optional[List[ActionObjectPermission]] = None,
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
    ) -> Result[Optional[QueueItem], str]:
        qks = QueryKeys(qks=[UIDPartitionKey.with_obj(uid)])
        item = self.query_one(credentials=credentials, qks=qks)
        return item

    def pop(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[Optional[QueueItem], str]:
        item = self.get_by_uid(credentials=credentials, uid=uid)
        self.delete_by_uid(credentials=credentials, uid=uid)
        return item

    def pop_on_complete(
        self, credentials: SyftVerifyKey, uid: UID
    ) -> Result[Optional[QueueItem], str]:
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
