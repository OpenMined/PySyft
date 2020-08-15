from typing import List
from enum import Enum

from syft.core.node.domain.service.accept_or_deny_request_service import (
    AcceptOrDenyRequestMessage,
)
from ..... import serialize, deserialize
from ....common import UID
from .....proto.core.node.domain.service.request_message_pb2 import (
    RequestMessage as RequestMessage_PB,
)
from ....io.address import Address
from ....common.message import ImmediateSyftMessageWithoutReply
from ...common.service.node_service import ImmediateNodeServiceWithoutReply
from .....decorators import syft_decorator
from ...abstract.node import AbstractNode
from ...abstract.node import AbstractNodeClient
from nacl.signing import VerifyKey


class RequestStatus(Enum):
    Pending = 1
    Rejected = 2
    Accepted = 3


class RequestMessage(ImmediateSyftMessageWithoutReply):

    __slots__ = ["request_name", "request_description", "request_id"]

    def __init__(
        self,
        object_id: UID,
        address: Address,
        requester_verify_key: VerifyKey,
        owner_address: Address,
        request_name: str = "",
        request_description: str = "",
        request_id: UID = UID(),
        owner_client_if_available: AbstractNodeClient = None,
    ):
        super().__init__(address=address, msg_id=request_id)
        self.request_name = request_name
        self.request_description = request_description
        self.request_id = request_id
        self.requester_verify_key = requester_verify_key
        self.object_id = object_id
        self.owner_address = owner_address
        self.owner_client_if_available = owner_client_if_available

    def accept(self):
        msg = AcceptOrDenyRequestMessage(
            address=self.owner_client_if_available.address,
            accept=True,
            request_id=self.id,
        )
        self.owner_client_if_available.send_immediate_msg_without_reply(msg=msg)

    def deny(self):
        msg = AcceptOrDenyRequestMessage(
            address=self.owner_client_if_available.address,
            accept=False,
            request_id=self.id,
        )
        self.owner_client_if_available.send_immediate_msg_without_reply(msg=msg)

    def withdraw(self):
        self.deny()

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> RequestMessage_PB:
        msg = RequestMessage_PB()
        msg.request_name = self.request_name
        msg.request_description = self.request_description
        msg.request_id.CopyFrom(serialize(obj=self.request_id))
        msg.target_address.CopyFrom(serialize(obj=self.address))
        msg.object_id.CopyFrom(serialize(obj=self.object_id))
        msg.owner_address.CopyFrom(serialize(obj=self.owner_address))
        msg.requester_verify_key = bytes(self.requester_verify_key)
        return msg

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: RequestMessage_PB) -> "RequestMessage":
        request_msg = RequestMessage(
            request_id=deserialize(blob=proto.request_id),
            request_name=proto.request_name,
            request_description=proto.request_description,
            address=deserialize(blob=proto.target_address),
            object_id=deserialize(blob=proto.object_id),
            owner_address=deserialize(blob=proto.owner_address),
            requester_verify_key=VerifyKey(proto.requester_verify_key),
        )
        request_msg.request_id = deserialize(blob=proto.request_id)
        return request_msg

    @staticmethod
    @syft_decorator(typechecking=True)
    def get_protobuf_schema() -> type:
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
            if req.id == msg.id:
                raise Exception("This request already exists!")

        node.requests.append(msg)
