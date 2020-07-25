"""In this class, we support the functionality necessary to support
virtual network connections between nodes in the Syft ecosystem.
Replacing this object with an actual network connection object
(such as one powered by P2P tech, web sockets, or HTTP) should
execute the exact same functionality but do so over a network"""

from ..message.syft_message import SyftMessageWithReply
from ..message.syft_message import SyftMessageWithoutReply
from ..nodes.abstract.node import Node
from ...decorators import syft_decorator
from typing import final

from .abstract import ServerConnection
from .abstract import ClientConnection

known_objects = {}


@final
class VirtualServerConnection(ServerConnection):
    @syft_decorator(typechecking=True)
    def __init__(self, node: Node):
        self.node = node

    @syft_decorator(typechecking=True)
    def recv_msg_with_reply(self, msg: SyftMessageWithReply) -> SyftMessageWithoutReply:
        return self.node.recv_msg_with_reply(msg=msg)

    @syft_decorator(typechecking=True)
    def recv_msg_without_reply(self, msg: SyftMessageWithoutReply) -> None:
        self.node.recv_msg_without_reply(msg=msg)


@final
class VirtualClientConnection(ClientConnection):
    @syft_decorator(typechecking=True)
    def __init__(self, server: VirtualServerConnection):
        self.server = server

    def send_msg_without_reply(self, msg: SyftMessageWithoutReply) -> None:
        self.server.recv_msg_without_reply(msg=msg)

    @syft_decorator(typechecking=True)
    def send_msg_with_reply(self, msg: SyftMessageWithReply) -> SyftMessageWithoutReply:
        return self.server.recv_msg_with_reply(msg=msg)


@syft_decorator(typechecking=True)
def create_virtual_connection(node: Node) -> VirtualClientConnection:

    server = VirtualServerConnection(node=node)
    client = VirtualClientConnection(server=server)

    return client
