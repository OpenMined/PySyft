# stdlib
from datetime import date
from io import StringIO
import sys
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import VerifyKey
from pydantic import EmailStr
from typing_extensions import final

# relative
from ......core.common.uid import UID
from .....common.serde.serializable import serializable
from ....domain_interface import DomainInterface
from ....domain_msg_registry import DomainMessageRegistry
from ...node_table.utils import model_to_json
from ...permissions.permissions import BasePermission
from ...permissions.user_permissions import NoRestriction
from ...permissions.user_permissions import UserIsOwner
from ..generic_payload.syft_message import NewSyftMessage as SyftMessage
from ..generic_payload.syft_message import ReplyPayload
from ..generic_payload.syft_message import RequestPayload


def run_task(code: str) -> None:
    # create file-like string to capture output
    codeOut = StringIO()
    codeErr = StringIO()

    sys.stdout = codeOut
    sys.stderr = codeErr
    exec(code)
    # restore stdout and stderr
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print("Error: ", codeErr.getvalue())
    print("Std Output: ", codeOut.getvalue())


@serializable(recursive_serde=True)
@final
class CreateTask(SyftMessage, DomainMessageRegistry):

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        """Payload fields and types used during a User Creation Request."""

        code: str

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        """Payload fields and types used during a User Creation Response."""

        message: str = "Your task was successfully submited!"

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

        if not hasattr(node, "tasks"):
            node.tasks = {}

        if not node.tasks.get(user.id, None):
            node.tasks[user.id] = []

        user_role = node.roles.first(id=user.role).name
        if user_role:
            # Mock, replace it by database calls
            node.tasks[user.id].append(
                {
                    "id": UID().to_string(),
                    "user": {"name": user.name, "role": user_role, "email": user.email},
                    "code": self.payload.code,
                    "status": "pending",
                    "created_at": date.today().strftime("%d/%m/%Y"),
                    "updated_at": "--",
                    "reviewed_by": "--",
                    "execution": "--",
                }
            )
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

        tasks = node.tasks[user.id]
        return GetTasks.Reply(tasks=tasks)

    def get_permissions(self) -> List[Type[BasePermission]]:
        """Returns the list of permission classes."""
        return [NoRestriction]


@serializable(recursive_serde=True)
@final
class ReviewTask(SyftMessage, DomainMessageRegistry):

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        task_id: str
        reason: str
        status: str

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        message: str = "Review submitted!"

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

        for user_tasks in node.tasks.values():
            for task in user_tasks:
                if task["id"] == self.payload.task_id:
                    task["status"] = status
                    task["reason"] = self.payload.reason
                    task["reviewed_by"] = user.name
                    task["updated_at"] = date.today().strftime("%d/%m/%Y")
                    if status == "accepted":
                        run_task(task["code"])
                        task["execution"] = {"status": "enqueued"}
        return CreateTask.Reply()

    def get_permissions(self) -> List[Type[BasePermission]]:
        """Returns the list of permission classes."""
        return [UserIsOwner]
