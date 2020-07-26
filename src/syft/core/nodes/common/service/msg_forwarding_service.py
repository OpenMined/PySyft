from __future__ import annotations

from .....decorators import syft_decorator
from .node_service import NodeServiceWithoutReply
from .node_service import NodeServiceWithReply
from ...abstract.node import AbstractNode
from ....message.syft_message import SyftMessageWithoutReply
from ....message.syft_message import SyftMessageWithReply
from typing import List


class MessageWithoutReplyForwardingService(NodeServiceWithoutReply):
    @syft_decorator(typechecking=True)
    def process(self, node: AbstractNode, msg: SyftMessageWithoutReply) -> None:

        addr = msg.address
        pri_addr = addr.pri_address
        pub_addr = addr.pub_address

        if pri_addr.vm is not None and node.store.has_object(pri_addr.vm):
            return node.store.get_object(pri_addr.vm).send_msg_without_reply(msg=msg)

        if pri_addr.device is not None and node.store.has_object(pri_addr.device):
            return node.store.get_object(pri_addr.device).send_msg_without_reply(msg=msg)

        if pub_addr.domain is not None and node.store.has_object(pub_addr.domain):
            return node.store.get_object(pub_addr.domain).send_msg_without_reply(msg=msg)

        if pub_addr.network is not None and node.store.has_object(pub_addr.network):
            return node.store.get_object(pub_addr.network).send_msg_without_reply(msg=msg)

        raise Exception("Address unknown - cannot forward message. Throwing it away.")

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [SyftMessageWithoutReply]


class MessageWithReplyForwardingService(NodeServiceWithReply):
    @syft_decorator(typechecking=True)
    def process(self, node: AbstractNode, msg: SyftMessageWithReply) -> None:

        addr = msg.address
        pri_addr = addr.pri_address
        pub_addr = addr.pub_address

        if pri_addr.vm is not None and node.store.has_object(pri_addr.vm):
            return node.store.get_object(pri_addr.vm).send_msg_with_reply(msg=msg)

        if pri_addr.device is not None and node.store.has_object(pri_addr.device):
            return node.store.get_object(pri_addr.device).send_msg_with_reply(msg=msg)

        if pub_addr.domain is not None and node.store.has_object(pub_addr.domain):
            return node.store.get_object(pub_addr.domain).send_msg_with_reply(msg=msg)

        if pub_addr.network is not None and node.store.has_object(pub_addr.network):
            return node.store.get_object(pub_addr.network).send_msg_with_reply(msg=msg)

        raise Exception("Address unknown - cannot forward message. Throwing it away.")

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [SyftMessageWithReply]
