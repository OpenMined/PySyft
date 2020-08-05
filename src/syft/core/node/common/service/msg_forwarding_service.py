from typing import List

from syft.core.common.message import (
    ImmediateSyftMessageWithoutReply,
    ImmediateSyftMessageWithReply,
)

from .....decorators import syft_decorator
from ...abstract.node import AbstractNode
from .node_service import (
    ImmediateNodeServiceWithoutReply,
    ImmediateNodeServiceWithReply,
)


class MessageWithoutReplyForwardingService(ImmediateNodeServiceWithoutReply):
    @syft_decorator(typechecking=True)
    def process(
        self, node: AbstractNode, msg: ImmediateSyftMessageWithoutReply
    ) -> None:

        addr = msg.address
        pri_addr = addr.pri_address
        pub_addr = addr.pub_address

        if pri_addr.vm is not None and node.store.has_object(pri_addr.vm):
            return node.store.get_object(pri_addr.vm).send_immediate_msg_without_reply(
                msg=msg
            )

        if pri_addr.device is not None and node.store.has_object(pri_addr.device):

            return node.store.get_object(
                pri_addr.device
            ).send_immediate_msg_without_reply(msg=msg)

        if pub_addr.domain is not None and node.store.has_object(pub_addr.domain):
            return node.store.get_object(
                pub_addr.domain
            ).send_immediate_msg_without_reply(msg=msg)

        if pub_addr.network is not None and node.store.has_object(pub_addr.network):
            return node.store.get_object(
                pub_addr.network
            ).send_immediate_msg_without_reply(msg=msg)

        raise Exception(
            "Address unknown - cannot forward old_message. Throwing it away."
        )

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [ImmediateSyftMessageWithoutReply]


class MessageWithReplyForwardingService(ImmediateNodeServiceWithReply):
    @syft_decorator(typechecking=True)
    def process(
        self, node: AbstractNode, msg: ImmediateSyftMessageWithReply
    ) -> ImmediateSyftMessageWithoutReply:

        addr = msg.address
        pri_addr = addr.pri_address
        pub_addr = addr.pub_address

        if pri_addr.vm is not None and node.store.has_object(pri_addr.vm):
            return node.store.get_object(pri_addr.vm).send_immediate_msg_with_reply(
                msg=msg
            )

        if pri_addr.device is not None and node.store.has_object(pri_addr.device):
            return node.store.get_object(pri_addr.device).send_immediate_msg_with_reply(
                msg=msg
            )

        if pub_addr.domain is not None and node.store.has_object(pub_addr.domain):
            return node.store.get_object(pub_addr.domain).send_immediate_msg_with_reply(
                msg=msg
            )

        if pub_addr.network is not None and node.store.has_object(pub_addr.network):
            return node.store.get_object(
                pub_addr.network
            ).send_immediate_msg_with_reply(msg=msg)

        raise Exception(
            "Address unknown - cannot forward old_message. Throwing it away."
        )

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [ImmediateSyftMessageWithReply]
