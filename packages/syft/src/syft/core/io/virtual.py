"""In this class, we support the functionality necessary to support
virtual network connections between the nodes in the Syft ecosystem.
Replacing this object with an actual network connection object
(such as one powered by P2P tech, web sockets, or HTTP) should
execute the exact same functionality but do so over a network"""

# stdlib
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from typing_extensions import final

# relative
from ...proto.core.io.connection_pb2 import (
    VirtualClientConnection as VirtualClientConnection_PB,
)
from ...proto.core.io.connection_pb2 import (
    VirtualServerConnection as VirtualServerConnection_PB,
)
from ..common.message import SignedEventualSyftMessageWithoutReply
from ..common.message import SignedImmediateSyftMessageWithReply
from ..common.message import SignedImmediateSyftMessageWithoutReply
from ..common.serde.deserialize import _deserialize
from ..node.abstract.node import AbstractNode
from .connection import ClientConnection
from .connection import ServerConnection


@final
class VirtualServerConnection(ServerConnection):
    def __init__(self, node: AbstractNode):
        self.node = node

    def recv_immediate_msg_with_reply(
        self, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithoutReply:
        return self.node.recv_immediate_msg_with_reply(msg=msg)

    def recv_immediate_msg_without_reply(
        self, msg: SignedImmediateSyftMessageWithoutReply
    ) -> None:
        self.node.recv_immediate_msg_without_reply(msg=msg)

    def recv_eventual_msg_without_reply(
        self, msg: SignedEventualSyftMessageWithoutReply
    ) -> None:
        self.node.recv_eventual_msg_without_reply(msg=msg)

    def _object2proto(self) -> VirtualServerConnection_PB:
        return VirtualServerConnection_PB(node=self.node._object2proto())

    @staticmethod
    def _proto2object(proto: VirtualServerConnection_PB) -> "VirtualServerConnection":
        node = _deserialize(blob=proto.node, from_proto=True)
        return VirtualServerConnection(
            node=node,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return VirtualServerConnection_PB


@final
class VirtualClientConnection(ClientConnection):
    def __init__(self, server: VirtualServerConnection):
        self.server = server

    def send_immediate_msg_without_reply(
        self,
        msg: SignedImmediateSyftMessageWithoutReply,
        timeout: Optional[float] = None,
    ) -> None:
        self.server.recv_immediate_msg_without_reply(msg=msg)

    def send_immediate_msg_with_reply(
        self,
        msg: SignedImmediateSyftMessageWithReply,
        timeout: Optional[float] = None,
        return_signed: bool = False,
    ) -> SignedImmediateSyftMessageWithoutReply:
        return self.server.recv_immediate_msg_with_reply(msg=msg)

    def send_eventual_msg_without_reply(
        self,
        msg: SignedEventualSyftMessageWithoutReply,
        timeout: Optional[float] = None,
    ) -> None:
        return self.server.recv_eventual_msg_without_reply(msg=msg)

    def _object2proto(self) -> VirtualClientConnection_PB:
        return VirtualClientConnection_PB(server=self.server._object2proto())

    @staticmethod
    def _proto2object(proto: VirtualClientConnection_PB) -> "VirtualClientConnection":
        return VirtualClientConnection(
            server=VirtualServerConnection._proto2object(proto.server),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return VirtualClientConnection_PB


def create_virtual_connection(node: AbstractNode) -> VirtualClientConnection:

    server = VirtualServerConnection(node=node)
    client = VirtualClientConnection(server=server)

    return client
