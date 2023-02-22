# stdlib
from datetime import datetime
from enum import Enum
import hashlib
from typing import Callable
from typing import List
from typing import Optional

# third party
from result import Err
from result import Ok
from result import Result
from typing_extensions import Self

# relative
from ....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from ....core.node.common.node_table.syft_object import SyftBaseObject
from ....core.node.common.node_table.syft_object import SyftObject
from ...common.serde import _serialize
from ...common.serde.serializable import serializable
from ...common.uid import UID
from .action_service import ActionService
from .action_store import ActionObjectPermission
from .action_store import ActionPermission
from .context import AuthedServiceContext
from .credentials import SyftVerifyKey
from .node import NewNode
from .response import SyftError
from .response import SyftSuccess
from .transforms import TransformContext
from .transforms import add_credentials_for_key
from .transforms import add_node_uid_for_key
from .transforms import generate_id
from .transforms import transform


@serializable(recursive_serde=True)
class DateTime(SyftObject):
    __canonical_name__ = "DateTime"
    __version__ = SYFT_OBJECT_VERSION_1

    utc_timestamp: float

    @staticmethod
    def now() -> Self:
        return DateTime(utc_timestamp=datetime.utcnow().timestamp())

    def __str__(self) -> str:
        utc_datetime = datetime.utcfromtimestamp(self.utc_timestamp)
        return utc_datetime.strftime("%Y-%m-%d %H:%M:%S")


@serializable(recursive_serde=True)
class RequestStatus(Enum):
    PENDING = 0
    REJECTED = 1
    APPROVED = 2


class ChangeContext(SyftBaseObject):
    node: Optional[NewNode] = None
    approving_user_credentials: Optional[SyftVerifyKey]
    requesting_user_credentials: Optional[SyftVerifyKey]

    @staticmethod
    def from_service(context: AuthedServiceContext) -> Self:
        return ChangeContext(
            node=context.node, approving_user_credentials=context.credentials
        )


class Change(SyftObject):
    __canonical_name__ = "Change"
    __version__ = SYFT_OBJECT_VERSION_1


@serializable(recursive_serde=True)
class ActionStoreChange(Change):
    __canonical_name__ = "ActionStoreChange"
    __version__ = SYFT_OBJECT_VERSION_1

    action_object_uid: UID
    apply_permission_type: ActionPermission

    def _run(
        self, context: ChangeContext, apply: bool
    ) -> Result[SyftSuccess, SyftError]:
        try:
            action_service = context.node.get_service(ActionService)
            action_store = action_service.store
            owner_permission = ActionObjectPermission(
                uid=self.action_object_uid,
                credentials=context.approving_user_credentials,
                permission=self.apply_permission_type,
            )
            if action_store.has_permission(permission=owner_permission):
                requesting_permission = ActionObjectPermission(
                    uid=self.action_object_uid,
                    credentials=context.requesting_user_credentials,
                    permission=self.apply_permission_type,
                )
                if apply:
                    action_store.add_permission(requesting_permission)
                else:
                    action_store.remove_permission(requesting_permission)
            else:
                return Err(
                    SyftError(
                        message=f"No permission for approving_user_credentials {context.approving_user_credentials}"
                    )
                )
            return Ok(SyftSuccess(message=f"{type(self)} Success"))
        except Exception as e:
            print(f"failed to apply {type(self)}")
            return Err(SyftError(message=e))

    def apply(self, context: ChangeContext) -> Result[SyftSuccess, SyftError]:
        return self._run(context=context, apply=True)

    def revert(self, context: ChangeContext) -> Result[SyftSuccess, SyftError]:
        return self._run(context=context, apply=False)


@serializable(recursive_serde=True)
class Request(SyftObject):
    __canonical_name__ = "Request"
    __version__ = SYFT_OBJECT_VERSION_1

    requesting_user_verify_key: SyftVerifyKey
    approving_user_verify_key: Optional[SyftVerifyKey]
    request_time: DateTime
    approval_time: Optional[DateTime]
    status: RequestStatus = RequestStatus.PENDING
    node_uid: UID
    request_hash: str
    changes: List[Change]

    __attr_searchable__ = [
        "requesting_user_verify_key",
        "approving_user_verify_key",
        "status",
    ]
    __attr_unique__ = ["request_hash"]

    def apply(self, context: AuthedServiceContext) -> Result[SyftSuccess, SyftError]:
        change_context = ChangeContext.from_service(context)
        change_context.requesting_user_credentials = self.requesting_user_verify_key
        for change in self.changes:
            result = change.apply(context=change_context)
            if result.is_err():
                return result
        return Ok(SyftSuccess(message=f"Request {self.id} changes applied"))

    def revert(self, context: AuthedServiceContext) -> Result[SyftSuccess, SyftError]:
        change_context = ChangeContext.from_service(context)
        change_context.requesting_user_credentials = self.requesting_user_verify_key
        for change in self.changes:
            result = change.revert(context=change_context)
            if result.is_err():
                return result
        return Ok(SyftSuccess(message=f"Request {self.id} changes reverted"))


@serializable(recursive_serde=True)
class SubmitRequest(SyftObject):
    __canonical_name__ = "SubmitRequest"
    __version__ = SYFT_OBJECT_VERSION_1

    changes: List[Change]


def hash_changes(context: TransformContext) -> TransformContext:
    request_time = context.output["request_time"]
    key = context.output["requesting_user_verify_key"]
    changes = context.output["changes"]

    time_hash = hashlib.sha256(
        _serialize(request_time.utc_timestamp, to_bytes=True)
    ).digest()
    key_hash = hashlib.sha256(bytes(key.verify_key)).digest()
    changes_hash = hashlib.sha256(_serialize(changes, to_bytes=True)).digest()
    final_hash = hashlib.sha256((time_hash + key_hash + changes_hash)).hexdigest()

    context.output["request_hash"] = final_hash
    return context


def add_request_time(context: TransformContext) -> TransformContext:
    context.output["request_time"] = DateTime.now()
    return context


@transform(SubmitRequest, Request)
def submit_request_to_request() -> List[Callable]:
    return [
        generate_id,
        add_node_uid_for_key("node_uid"),
        add_request_time,
        add_credentials_for_key("requesting_user_verify_key"),
        hash_changes,
    ]
