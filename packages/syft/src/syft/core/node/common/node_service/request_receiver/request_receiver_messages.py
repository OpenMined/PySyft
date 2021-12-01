# stdlib
from enum import Enum
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# relative
from ...... import deserialize
from ...... import serialize
from ......logger import critical
from ......logger import debug
from ......logger import traceback
from ......logger import traceback_and_raise
from ......proto.core.node.domain.service.request_message_pb2 import (
    RequestMessage as RequestMessage_PB,
)
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
@serializable()
class RequestMessage(ImmediateSyftMessageWithoutReply):

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

    def _object2proto(self) -> RequestMessage_PB:
        msg = RequestMessage_PB()
        msg.object_tags.extend(self.object_tags)
        msg.object_type = self.object_type
        msg.status = self.status
        msg.request_type = self.request_type
        msg.date = self.date
        msg.user_name = self.user_name
        msg.user_email = self.user_email
        msg.user_role = self.user_role
        msg.requested_budget = self.requested_budget if self.requested_budget else 0.0
        msg.current_budget = self.current_budget if self.current_budget else 0.0
        msg.request_description = self.request_description
        msg.request_id.CopyFrom(serialize(obj=self.request_id))
        msg.target_address.CopyFrom(serialize(obj=self.address))
        msg.object_id.CopyFrom(serialize(obj=self.object_id))
        msg.owner_address.CopyFrom(serialize(obj=self.owner_address))
        msg.requester_verify_key = bytes(self.requester_verify_key)

        # -1 will represent no timeout, where as 0 is a valid value for timing out
        # immediately after checking if there is a rule in place to accept or deny
        if self.timeout_secs is None or self.timeout_secs < 0:
            self.timeout_secs = -1
        msg.timeout_secs = int(self.timeout_secs)
        return msg

    @staticmethod
    def _proto2object(proto: RequestMessage_PB) -> "RequestMessage":
        request_msg = RequestMessage(
            request_id=deserialize(blob=proto.request_id),
            status=proto.status,
            request_type=proto.request_type,
            date=proto.date,
            object_tags=proto.object_tags,
            object_type=proto.object_type,
            user_name=proto.user_name,
            user_email=proto.user_email,
            user_role=proto.user_role,
            requested_budget=proto.requested_budget,
            current_budget=proto.current_budget,
            request_description=proto.request_description,
            address=deserialize(blob=proto.target_address),
            object_id=deserialize(blob=proto.object_id),
            owner_address=deserialize(blob=proto.owner_address),
            requester_verify_key=VerifyKey(proto.requester_verify_key),
            timeout_secs=proto.timeout_secs,
        )
        request_msg.request_id = deserialize(blob=proto.request_id)
        return request_msg

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return RequestMessage_PB
