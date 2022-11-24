# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
from nacl.signing import VerifyKey
from typing_extensions import final

# relative
from .....common.serde.serializable import serializable
from ....domain_interface import DomainInterface
from ....domain_msg_registry import DomainMessageRegistry
from ...node_table.utils import model_to_json
from ...permissions.permissions import BasePermission
from ...permissions.user_permissions import UserCanTriageRequest
from ..generic_payload.syft_message import NewSyftMessage as SyftMessage
from ..generic_payload.syft_message import ReplyPayload
from ..generic_payload.syft_message import RequestPayload
from ..request_receiver.request_receiver_messages import RequestStatus


@serializable(recursive_serde=True)
@final
class NewGetRequestMessage(SyftMessage, DomainMessageRegistry):

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        """Payload fields and types used during a GetRequest Request."""

        request_id: str

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        """Payload fields and types used during a GetRequest Response."""

        id: str
        date: str
        user_id: int
        user_name: str
        user_email: str = ""
        user_role: str = ""
        user_budget: float
        institution: Union[str, None] = ""
        website: Union[str, None] = ""
        object_id: Union[str, None]
        reason: Union[str, None] = ""
        status: str = "pending"
        request_type: str
        verify_key: str
        object_type: str = ""
        tags: List[str]
        updated_on: Union[str, None] = ""
        reviewer_name: Union[str, None] = ""
        reviewer_role: Union[str, None] = ""
        reviewer_comment: Union[str, None] = ""
        requested_budget: Union[float, None] = 0
        current_budget: Union[float, None] = 0

    request_payload_type = (
        Request  # Converts generic syft dict into a CreateUserMessage.Request object.
    )
    reply_payload_type = (
        Reply  # Creates a proper Reply payload message structure as a response.
    )

    def run(  # type: ignore
        self, node: DomainInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:  # type: ignore
        # Get Payload Content

        request = node.data_requests.all()[0]

        request_json = model_to_json(request)
        print("My Request JSON: ", request_json)
        return NewGetRequestMessage.Reply(**request_json)

    def get_permissions(self) -> List[Type[BasePermission]]:
        """Returns the list of permission classes."""
        return [UserCanTriageRequest]


@serializable(recursive_serde=True)
@final
class NewGetDataRequestsMessage(SyftMessage, DomainMessageRegistry):

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        """Payload fields and types used during a GetRequests Request."""

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        """Payload fields and types used during a GetRequests Response."""

        requests: List[Dict[str, Any]] = []

    request_payload_type = (
        Request  # Converts generic syft dict into a CreateUserMessage.Request object.
    )
    reply_payload_type = (
        Reply  # Creates a proper Reply payload message structure as a response.
    )

    def run(  # type: ignore
        self, node: DomainInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:  # type: ignore
        requests = node.data_requests.query(request_type="data")
        response = list()
        for request in requests:
            # Get current state user
            if node.data_requests.status(request.id) == RequestStatus.Pending:
                _user = node.users.first(id=request.user_id)
                user = model_to_json(_user)
                user["role"] = node.roles.first(id=_user.role).name
                user["current_budget"] = user["budget"]
            # Get History state user
            else:
                user = node.data_requests.get_user_info(request_id=request.id)
            request = model_to_json(request)
            response.append({"user": user, "req": request})
        return NewGetDataRequestsMessage.Reply(requests=response)

    def get_permissions(self) -> List[Type[BasePermission]]:
        """Returns the list of permission classes."""
        return [UserCanTriageRequest]


@serializable(recursive_serde=True)
@final
class NewGetBudgetRequestsMessage(SyftMessage, DomainMessageRegistry):

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        """Payload fields and types used during a GetRequests Request."""

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        """Payload fields and types used during a GetRequests Response."""

        requests: List[Dict[str, Any]] = []

    request_payload_type = (
        Request  # Converts generic syft dict into a CreateUserMessage.Request object.
    )
    reply_payload_type = (
        Reply  # Creates a proper Reply payload message structure as a response.
    )

    def run(  # type: ignore
        self, node: DomainInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:  # type: ignore
        response = list()
        requests = node.data_requests.query(request_type="budget")
        for request in requests:
            # Get current state user
            if node.data_requests.status(request.id) == RequestStatus.Pending:
                _user = node.users.first(id=request.user_id)
                user = model_to_json(_user)
                user["role"] = node.roles.first(id=_user.role).name
                user["current_budget"] = user["budget"]
                request = model_to_json(request)
            # Get History state user
            else:
                user = node.data_requests.get_user_info(request_id=request.id)
                request = node.data_requests.get_req_info(request_id=request.id)
            response.append({"user": user, "req": request})
        return NewGetBudgetRequestsMessage.Reply(requests=response)

    def get_permissions(self) -> List[Type[BasePermission]]:
        """Returns the list of permission classes."""
        return [UserCanTriageRequest]


@serializable(recursive_serde=True)
@final
class NewUpdateRequestsMessage(SyftMessage, DomainMessageRegistry):

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        """Payload fields and types used during a GetRequests Request."""

        request_id: str
        status: str

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        """Payload fields and types used during a GetRequests Response."""

        requests: List[Dict[str, Any]] = []

    request_payload_type = (
        Request  # Converts generic syft dict into a CreateUserMessage.Request object.
    )
    reply_payload_type = (
        Reply  # Creates a proper Reply payload message structure as a response.
    )

    def run(  # type: ignore
        self, node: DomainInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:  # type: ignore
        response = list()
        requests = node.data_requests.query(request_type="budget")
        for request in requests:
            # Get current state user
            if node.data_requests.status(request.id) == RequestStatus.Pending:
                _user = node.users.first(id=request.user_id)
                user = model_to_json(_user)
                user["role"] = node.roles.first(id=_user.role).name
                user["current_budget"] = user["budget"]
                request = model_to_json(request)
            # Get History state user
            else:
                user = node.data_requests.get_user_info(request_id=request.id)
                request = node.data_requests.get_req_info(request_id=request.id)
            response.append({"user": user, "req": request})
        return NewGetBudgetRequestsMessage.Reply(requests=response)

    def get_permissions(self) -> List[Type[BasePermission]]:
        """Returns the list of permission classes."""
        return [UserCanTriageRequest]
