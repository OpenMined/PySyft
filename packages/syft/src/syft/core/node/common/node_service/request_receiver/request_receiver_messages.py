# stdlib
from enum import Enum
from typing import List
from typing import Optional

# third party
from nacl.signing import VerifyKey

# relative
from ......logger import critical
from ......logger import debug
from ......logger import traceback
from ......logger import traceback_and_raise
from .....common import UID
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.serializable import serializable
from .....io.address import Address
from ...client import Client
from ...node import Node
from ..accept_or_deny_request.accept_or_deny_request_messages import (
    AcceptOrDenyRequestMessage,
)


class RequestStatus(Enum):
    Pending = 1
    Rejected = 2
    Accepted = 3


# TODO: this message conflates Message functionality with Manager/Request_API functionality
# TODO: this needs to be split into two separate pieces of functionality.
@serializable(recursive_serde=True)
class RequestMessage(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = [
        "_id",
        "object_tags",
        "status",
        "request_type",
        "date",
        "user_name",
        "user_email",
        "user_role",
        "requested_budget",
        "current_budget",
        "request_description",
        "request_id",
        "address",
        "object_id",
        "owner_address",
        "requester_verify_key",
        "timeout_secs",
    ]
    __slots__ = ["name", "request_description", "request_id"]

    def __init__(
        self,
        object_id: UID,
        address: Address,
        requester_verify_key: VerifyKey,
        owner_address: Address,
        status: Optional[str] = "",
        request_type: Optional[str] = "",
        date: Optional[str] = "",
        object_tags: Optional[List[str]] = None,
        object_type: str = "",
        request_description: str = "",
        request_id: Optional[UID] = None,
        owner_client_if_available: Optional[Client] = None,
        destination_node_if_available: Optional[Node] = None,
        timeout_secs: Optional[int] = None,
        requested_budget: Optional[float] = 0.0,
        current_budget: Optional[float] = 0.0,
        user_name: Optional[str] = "",
        user_role: Optional[str] = "",
        user_email: Optional[str] = "",
    ):
        if request_id is None:
            request_id = UID()
        super().__init__(address=address, msg_id=request_id)

        self.object_tags = object_tags if object_tags else []
        self.object_type = object_type
        self.request_description = request_description
        self.request_id = request_id
        self.requester_verify_key = requester_verify_key
        self.object_id = object_id
        self.owner_address = owner_address
        self.owner_client_if_available = owner_client_if_available
        self.destination_node_if_available = destination_node_if_available
        self.timeout_secs = timeout_secs
        self._arrival_time: Optional[float] = None
        self.status: Optional[str] = status
        self.date: Optional[str] = date
        self.request_type: Optional[str] = request_type
        self.user_name: Optional[str] = user_name
        self.user_email: Optional[str] = user_email
        self.user_role: Optional[str] = user_role
        self.requested_budget: float = requested_budget  # type: ignore
        self.current_budget: float = current_budget  # type: ignore

    def accept(self) -> None:
        self.send_msg(accept=True)

    def approve(self) -> None:
        self.accept()

    def grant(self) -> None:
        self.accept()

    def deny(self) -> None:
        self.send_msg(accept=False)

    def send_msg(self, accept: bool) -> None:
        action_name = "Accept" if accept else "Deny"
        if self.owner_client_if_available is not None:
            msg = AcceptOrDenyRequestMessage(
                address=self.owner_client_if_available.address,
                accept=accept,
                request_id=self.id,
            )
            self.owner_client_if_available.send_immediate_msg_without_reply(msg=msg)
        elif self.destination_node_if_available is not None:
            msg = AcceptOrDenyRequestMessage(
                address=self.destination_node_if_available.address,
                accept=accept,
                request_id=self.id,
            )
            try:
                node = self.destination_node_if_available
                router = node.immediate_msg_without_reply_router
                service = router[type(msg)]
                service.process(
                    node=self.destination_node_if_available,
                    msg=msg,
                    verify_key=self.destination_node_if_available.root_verify_key,
                )
            except Exception as e:
                traceback(e)
                critical(f"Tried to {action_name} Message on Node. {e}")
            debug(f"{action_name} Request: " + str(self.id))
        else:
            log = f"No way to dispatch {action_name} Message."
            critical(log)
            traceback_and_raise(Exception(log))

    def reject(self) -> None:
        self.deny()

    def withdraw(self) -> None:
        self.deny()

    @property
    def arrival_time(self) -> Optional[float]:
        return self._arrival_time

    def set_arrival_time(self, arrival_time: float) -> None:
        # used to expire requests as their destination, this should never be serialized
        if self._arrival_time is None:
            self._arrival_time = arrival_time
