"""In this class, we support the functionality necessary to support
virtual network connections between nodes in the Syft ecosystem.
Replacing this object with an actual network connection object
(such as one powered by P2P tech, web sockets, or HTTP) should
execute the exact same functionality but do so over a network"""

from syft.core.common.message import ImmediateSyftMessageWithReply
from syft.core.common.message import ImmediateSyftMessageWithoutReply
from syft.core.common.message import EventualSyftMessageWithoutReply
from ..nodes.abstract.node import AbstractNode
from ...decorators import syft_decorator
from typing import final

from .connection import ServerConnection
from .connection import ClientConnection

known_objects = {}


@final
class VirtualServerConnection(ServerConnection):
    @syft_decorator(typechecking=True)
    def __init__(self, node: AbstractNode):
        self.node = node

    @syft_decorator(typechecking=True)
    def recv_immediate_msg_with_reply(
        self, msg: ImmediateSyftMessageWithReply
    ) -> ImmediateSyftMessageWithoutReply:
        return self.node.recv_immediate_msg_with_reply(msg=msg)

    @syft_decorator(typechecking=True)
    def recv_immediate_msg_without_reply(
        self, msg: ImmediateSyftMessageWithoutReply
    ) -> None:
        self.node.recv_immediate_msg_without_reply(msg=msg)

    @syft_decorator(typechecking=True)
    def recv_eventual_msg_without_reply(
        self, msg: EventualSyftMessageWithoutReply
    ) -> None:
        self.node.recv_eventual_msg_without_reply(msg=msg)


@final
class VirtualClientConnection(ClientConnection):
    @syft_decorator(typechecking=True)
    def __init__(self, server: VirtualServerConnection):
        self.server = server

    def send_immediate_msg_without_reply(
        self, msg: ImmediateSyftMessageWithoutReply
    ) -> None:
        self.server.recv_immediate_msg_without_reply(msg=msg)

    @syft_decorator(typechecking=True)
    def send_immediate_msg_with_reply(
        self, msg: ImmediateSyftMessageWithReply
    ) -> ImmediateSyftMessageWithoutReply:
        return self.server.recv_immediate_msg_with_reply(msg=msg)

    @syft_decorator(typechecking=True)
    def send_eventual_msg_without_reply(
        self, msg: EventualSyftMessageWithoutReply
    ) -> None:
        return self.server.recv_eventual_msg_without_reply(msg=msg)


@syft_decorator(typechecking=True)
def create_virtual_connection(node: AbstractNode) -> VirtualClientConnection:

    server = VirtualServerConnection(node=node)
    client = VirtualClientConnection(server=server)

    return client
