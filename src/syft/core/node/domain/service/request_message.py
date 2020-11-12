# stdlib
from enum import Enum
import time
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from loguru import logger
from nacl.signing import VerifyKey

# syft relative
from ..... import deserialize
from ..... import serialize
from .....decorators import syft_decorator
from .....proto.core.node.domain.service.request_message_pb2 import (
    RequestMessage as RequestMessage_PB,
)
from ....common import UID
from ....common.message import ImmediateSyftMessageWithoutReply
from ....io.address import Address
from ....node.common.client import Client
from ....node.common.node import DuplicateRequestException
from ...abstract.node import AbstractNode
from ...common.node import Node
from ...common.service.node_service import ImmediateNodeServiceWithoutReply
from ...domain.service.accept_or_deny_request_service import AcceptOrDenyRequestMessage


class RequestStatus(Enum):
    Pending = 1
    Rejected = 2
    Accepted = 3


class RequestMessage(ImmediateSyftMessageWithoutReply):

    __slots__ = ["name", "request_description", "request_id"]

    def __init__(
        self,
        object_id: UID,
        address: Address,
        requester_verify_key: VerifyKey,
        owner_address: Address,
        name: str = "",
        request_description: str = "",
        request_id: Optional[UID] = None,
        owner_client_if_available: Optional[Client] = None,
        destination_node_if_available: Optional[Node] = None,
        timeout_secs: Optional[int] = None,
    ):
        if request_id is None:
            request_id = UID()
        super().__init__(address=address, msg_id=request_id)
        self.name = name
        self.request_description = request_description
        self.request_id = request_id
        self.requester_verify_key = requester_verify_key
        self.object_id = object_id
        self.owner_address = owner_address
        self.owner_client_if_available = owner_client_if_available
        self.destination_node_if_available = destination_node_if_available
        self.timeout_secs = timeout_secs
        self._arrival_time: Optional[float] = None

    def accept(self) -> None:
        self.send_msg(accept=True)

    def approve(self) -> None:
        self.accept()

    def grant(self) -> None:
        self.accept()

    def deny(self) -> None:
        self.send_msg(accept=False)

    @syft_decorator(typechecking=True)
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
                print(e)
                logger.critical(f"Tried to {action_name} Message on Node. {e}")
            logger.debug(f"{action_name} Request: " + str(self.id))
        else:
            log = f"No way to dispatch {action_name} Message."
            logger.critical(log)
            raise Exception(log)

    def reject(self) -> None:
        self.deny()

    def withdraw(self) -> None:
        self.deny()

    @property
    def arrival_time(self) -> Optional[float]:
        return self._arrival_time

    @syft_decorator(typechecking=True)
    def set_arrival_time(self, arrival_time: float) -> None:
        # used to expire requests as their destination, this should never be serialized
        if self._arrival_time is None:
            self._arrival_time = arrival_time

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> RequestMessage_PB:
        msg = RequestMessage_PB()
        msg.name = self.name
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
    @syft_decorator(typechecking=True)
    def _proto2object(proto: RequestMessage_PB) -> "RequestMessage":
        request_msg = RequestMessage(
            request_id=deserialize(blob=proto.request_id),
            name=proto.name,
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
    @syft_decorator(typechecking=True)
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return RequestMessage_PB


class RequestService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [RequestMessage]

    @staticmethod
    @syft_decorator(typechecking=True)
    def process(node: AbstractNode, msg: RequestMessage, verify_key: VerifyKey) -> None:
        # node.requests.register_request(msg)  # type: ignore

        if msg.requester_verify_key != verify_key:
            raise Exception(
                "You tried to request access for a key that is not yours!"
                "You cannot do this! Whatever key you want to request access"
                "for must be the verify key that also verifies the message"
                "containing the request."
            )

        # since we reject/accept requests based on the ID, we don't want there to be
        # multiple requests with the same ID because this could cause security problems.
        for req in node.requests:
            # the same user has requested the same object so we raise a
            # DuplicateRequestException
            if (
                req.object_id == msg.object_id
                and req.requester_verify_key == msg.requester_verify_key
            ):
                raise DuplicateRequestException(
                    f"You have already requested {msg.object_id}"
                )

        # using the local arrival time we can expire the request
        msg.set_arrival_time(arrival_time=time.time())
        node.requests.append(msg)
