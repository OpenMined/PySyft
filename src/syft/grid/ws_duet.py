from typing import Optional

from syft.core.common.uid import UID
from syft.core.io.route import SoloRoute
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.domain.client import DomainClient
from syft.core.node.domain.domain import Domain
from syft.grid.connections.websocket_connection import WebsocketConnection

from nacl.signing import SigningKey


from syft.core.common.message import (
    SyftMessage,
    EventualSyftMessageWithoutReply,
    ImmediateSyftMessageWithoutReply,
    ImmediateSyftMessageWithReply,
)


class WSDuet(DomainClient):
    def __init__(
        self, url: str, node: Optional[AbstractNode] = None, name: Optional[str] = None
    ):
        # Duet Demo using Websocket/WebRTC approach.

        # To act as a server we need to bind our
        # websocket connection with a node type instance
        # that way, we'll be able to process requests
        # sent by other peers.
        # Ideally, this obj instance should be provided beforehand
        # to allow this process to handle with multiple
        # connections/peers using the same domain / data (concert?)
        # But if you want to keep different domains for different
        # peers. this option allow us to create a new domain for
        # each new connection.
        if not node:
            # generate a signing key
            self.signing_key = SigningKey.generate()
            self.verify_key = self.signing_key.verify_key
            self.node = Domain(name=name, root_key=self.verify_key)
        else:
            self.node = node

        # Create WebSocketConnection Instance
        self.conn = WebsocketConnection(url, self.node)

        address, name, client_id = self.conn.metadata
        route = SoloRoute(destination=client_id, connection=self.conn)

        # Update DomainClient metadata
        super().__init__(domain=address, name=name, routes=[route])

    @property
    def id(self) -> UID:
        return self.target_id

    def stop(self) -> None:
        self.__del__()

    def __del__(self) -> None:
        self.conn.connected = False

    def send_immediate_msg_with_reply(
        self, msg: ImmediateSyftMessageWithReply
    ) -> SyftMessage:
        return self.conn.send_immediate_msg_with_reply(msg=msg)

    def send_immediate_msg_without_reply(
        self, msg: ImmediateSyftMessageWithoutReply
    ) -> None:
        self.conn.send_immediate_msg_without_reply(msg=msg)

    def send_eventual_msg_without_reply(
        self, msg: EventualSyftMessageWithoutReply
    ) -> None:
        self.conn.send_eventual_msg_without_reply(msg=msg)
