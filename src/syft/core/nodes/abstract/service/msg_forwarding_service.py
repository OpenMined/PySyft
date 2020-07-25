from __future__ import annotations

from .....decorators import syft_decorator
from .node_service import NodeService
from ...common.node import AbstractNode
from .....common.message import AbstractMessage
from typing import List


class MessageForwardingService(NodeService):

    @syft_decorator(typechecking=True)
    def process(self, node: AbstractNode, msg: AbstractMessage) -> AbstractMessage:
        addr = msg.address
        pri_addr = addr.pri_address
        pub_addr = addr.pub_address
        if pri_addr.vm is not None and node.store.has_object(pri_addr.vm):
            return node.store.get_object(pri_addr.vm).send_msg(msg=msg)

        if pri_addr.device is not None and node.store.has_object(pri_addr.device):
            return node.store.get_object(pri_addr.device).send_msg(msg=msg)

        if pub_addr.domain is not None and node.store.has_object(pub_addr.domain):
            return node.store.get_object(pub_addr.domain).send_msg(msg=msg)

        if pub_addr.network is not None and node.store.has_object(pub_addr.network):
            return node.store.get_object(pub_addr.network).send_msg(msg=msg)

        raise Exception("Address unknown - cannot forward message. Throwing it away.")


    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [AbstractMessage]