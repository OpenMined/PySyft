"""In this class, we support the functionality necessary to support
virtual network connections between nodes in the Syft ecosystem.
Replacing this object with an actual network connection object
(such as one powered by P2P tech, web sockets, or HTTP) should
execute the exact same functionality but do so over a network"""

from ..message.syft_message import SyftMessage
from ..nodes.abstract.worker import Worker
from ...typecheck import type_hints
from typing import final

from .abstract import ServerConnection
from .abstract import ClientConnection

known_objects = {}


@final
class VirtualServerConnection(ServerConnection):

    @type_hints
    def __init__(self, worker:Worker):
        self.worker = worker

    @type_hints
    def recv_msg(self, msg:SyftMessage) -> SyftMessage:
        return self.worker.recv_msg(msg=msg)


@final
class VirtualClientConnection(ClientConnection):

    @type_hints
    def __init__(self, server:VirtualServerConnection):
        self.server = server

    @type_hints
    def send_msg(self, msg: SyftMessage) -> SyftMessage:
        return self.server.recv_msg(msg=msg)


@type_hints
def create_virtual_connection(worker:Worker) -> VirtualClientConnection:

    server = VirtualServerConnection(worker=worker)
    client = VirtualClientConnection(server=server)

    return client
