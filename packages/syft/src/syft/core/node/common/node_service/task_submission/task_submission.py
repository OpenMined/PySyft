# stdlib
from datetime import date
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

# third party
from nacl.signing import VerifyKey
from typing_extensions import final

# relative
from ......core.common.uid import UID
from .....common.serde.serializable import serializable
from ....domain_interface import DomainInterface
from ....domain_msg_registry import DomainMessageRegistry
from ...node_table.task import NoSQLTask
from ...permissions.permissions import BasePermission
from ...permissions.user_permissions import NoRestriction
from ...permissions.user_permissions import UserIsOwner
from ..generic_payload.syft_message import NewSyftMessage as SyftMessage
from ..generic_payload.syft_message import ReplyPayload
from ..generic_payload.syft_message import RequestPayload
from .enum import EXECUTION_STATUS
from .enum import TASK_SERVICE_DEFAULT_MESSAGES
from .enum import TASK_SERVICE_FIELDS
from .enum import TASK_SERVICE_STATUS


@serializable(recursive_serde=True)
@final
class CreateTask(SyftMessage, DomainMessageRegistry):

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        """Payload fields and types used during a User Creation Request."""

        inputs: Dict[str, str]
        code: str
        outputs: List[str]

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        """Payload fields and types used during a User Creation Response."""

        message: str = TASK_SERVICE_DEFAULT_MESSAGES.CREATE_TASK.value

    request_payload_type = (
        Request  # Converts generic syft dict into a CreateUserMessage.Request object.
    )
    reply_payload_type = (
        Reply  # Creates a proper Reply payload message structure as a response.
    )

    def run(  # type: ignore
        self, node: DomainInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:  # type: ignore
        user = node.users.get_user(verify_key=verify_key)

        task = NoSQLTask(
            uid=UID().to_string(),
            user=user.id.to_string(),
            inputs=self.payload.inputs,
            outputs={var: " -- " for var in self.payload.outputs},
            owner={"name": user.name, "role": user.role["name"], "email": user.email},
            code=self.payload.code,
            status=TASK_SERVICE_STATUS.PENDING.value,
            created_at=date.today().strftime("%d/%m/%Y"),
            updated_at=" -- ",
            reviewed_by=" -- ",
            execution={
                TASK_SERVICE_FIELDS.STATUS.value: EXECUTION_STATUS.WAITING.value
            },
        )
        node.tasks.add(task)
        return CreateTask.Reply()

    def get_permissions(self) -> List[Type[BasePermission]]:
        """Returns the list of permission classes."""
        return [NoRestriction]


@serializable(recursive_serde=True)
@final
class GetTasks(SyftMessage, DomainMessageRegistry):

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        pass

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        tasks: List[Dict[str, Any]]

    request_payload_type = (
        Request  # Converts generic syft dict into a CreateUserMessage.Request object.
    )
    reply_payload_type = (
        Reply  # Creates a proper Reply payload message structure as a response.
    )

    def run(  # type: ignore
        self, node: DomainInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:  # type: ignore
        user = node.users.get_user(verify_key=verify_key)

        if user.role["name"] == node.roles.owner_role["name"]:
            tasks = node.tasks.all()
        else:
            tasks = node.tasks.find(
                search_params={TASK_SERVICE_FIELDS.USER.value: user.id.to_string()}
            )
        return GetTasks.Reply(tasks=tasks)

    def get_permissions(self) -> List[Type[BasePermission]]:
        """Returns the list of permission classes."""
        return [NoRestriction]


@serializable(recursive_serde=True)
@final
class GetTask(SyftMessage, DomainMessageRegistry):

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        task_uid: str

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        code: str
        status: str
        owner: Dict[str, str]
        created_at: str
        updated_at: str
        reviewed_by: str
        execution: Dict[str, str]
        uid: str
        reason: str
        inputs: Dict[str, str] = {}
        outputs: Dict[str, str] = {}

    request_payload_type = (
        Request  # Converts generic syft dict into a CreateUserMessage.Request object.
    )
    reply_payload_type = (
        Reply  # Creates a proper Reply payload message structure as a response.
    )

    def run(  # type: ignore
        self, node: DomainInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:  # type: ignore
        user = node.users.get_user(verify_key=verify_key)
        if user.role["name"] == node.roles.owner_role["name"]:
            task = node.tasks.find_one(
                search_params={TASK_SERVICE_FIELDS.UID.value: self.payload.task_uid}
            )
        else:
            task = node.tasks.find_one(
                search_params={
                    TASK_SERVICE_FIELDS.USER.value: user.id.to_string(),
                    TASK_SERVICE_FIELDS.UID.value: self.payload.task_uid,
                }
            )

        return GetTask.Reply(**task)

    def get_permissions(self) -> List[Type[BasePermission]]:
        """Returns the list of permission classes."""
        return [NoRestriction]


@serializable(recursive_serde=True)
@final
class ReviewTask(SyftMessage, DomainMessageRegistry):

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        task_uid: str
        reason: str
        status: str

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        message: str = TASK_SERVICE_DEFAULT_MESSAGES.REVIEW_TASK.value

    request_payload_type = (
        Request  # Converts generic syft dict into a CreateUserMessage.Request object.
    )
    reply_payload_type = (
        Reply  # Creates a proper Reply payload message structure as a response.
    )

    def run(  # type: ignore
        self, node: DomainInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:  # type: ignore
        user = node.users.get_user(verify_key=verify_key)

        status = self.payload.status.lower()

        update_values = {
            TASK_SERVICE_FIELDS.STATUS.value: status,
            TASK_SERVICE_FIELDS.REASON.value: self.payload.reason,
            TASK_SERVICE_FIELDS.REVIEWED_BY.value: user.name,
            TASK_SERVICE_FIELDS.UPDATED_AT.value: date.today().strftime("%d/%m/%Y"),
            TASK_SERVICE_FIELDS.EXECUTION.value: {}
            if status != TASK_SERVICE_STATUS.ACCEPTED.value
            else {TASK_SERVICE_FIELDS.STATUS.value: EXECUTION_STATUS.ENQUEUED.value},
        }

        node.tasks.update(
            search_params={TASK_SERVICE_FIELDS.UID.value: self.payload.task_uid},
            updated_args=update_values,
        )

        return CreateTask.Reply()

    def get_permissions(self) -> List[Type[BasePermission]]:
        """Returns the list of permission classes."""
        return [UserIsOwner]
