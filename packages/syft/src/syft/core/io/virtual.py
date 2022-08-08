"""In this class, we support the functionality necessary to support
virtual network connections between the nodes in the Syft ecosystem.
Replacing this object with an actual network connection object
(such as one powered by P2P tech, web sockets, or HTTP) should
execute the exact same functionality but do so over a network"""

# stdlib
from typing import Optional

# third party
from typing_extensions import final

# relative
from ..common.message import SignedEventualSyftMessageWithoutReply
from ..common.message import SignedImmediateSyftMessageWithReply
from ..common.message import SignedImmediateSyftMessageWithoutReply
from ..common.serde.serializable import serializable
from ..node.abstract.node import AbstractNode
from .connection import ClientConnection
from .connection import ServerConnection


@final
@serializable(recursive_serde=True)
class VirtualServerConnection(ServerConnection):
    __attr_allowlist__ = ("node",)

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


@final
@serializable(recursive_serde=True)
class VirtualClientConnection(ClientConnection):
    __attr_allowlist__ = ("server",)

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


def create_virtual_connection(node: AbstractNode) -> VirtualClientConnection:
    server = VirtualServerConnection(node=node)
    client = VirtualClientConnection(server=server)

    return client
