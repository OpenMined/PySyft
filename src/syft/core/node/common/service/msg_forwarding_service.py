# external class imports
from typing import List

from syft.core.common.message import (
    ImmediateSyftMessageWithoutReply,
    ImmediateSyftMessageWithReply,
    SignedMessageT,
)

from .....decorators import syft_decorator
from ...abstract.node import AbstractNode
from .node_service import (
    SignedNodeServiceWithReply,
    SignedNodeServiceWithoutReply,
)


class SignedMessageWithoutReplyForwardingService(SignedNodeServiceWithoutReply):
    @staticmethod
    @syft_decorator(typechecking=True)
    def process(node: AbstractNode, msg: SignedMessageT) -> SignedMessageT:
        addr = msg.address
        # order is important, vm, device, domain, network
        for scope_id in [addr.vm_id, addr.device_id, addr.domain_id, addr.network_id]:
            if scope_id is not None and scope_id in node.store:
                obj = node.store[scope_id]
                try:
                    return obj.send_immediate_msg_without_reply(msg=msg)
                except Exception as e:
                    print(
                        f"{addr} in store doesnt have method send_immediate_msg_without_reply"
                    )
                    print(e)
                    pass

        try:
            in_memory_client = node.in_memory_client_registry[addr.target_id.id]
            return in_memory_client.send_immediate_msg_without_reply(msg=msg)
        except Exception as e:
            print(f"{addr} not on nodes in_memory_client. {e}")
            pass

        raise Exception(
            "Address unknown - cannot forward old_message. Throwing it away."
        )

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [ImmediateSyftMessageWithoutReply]


class SignedMessageWithReplyForwardingService(SignedNodeServiceWithReply):
    @staticmethod
    @syft_decorator(typechecking=True)
    def process(node: AbstractNode, msg: SignedMessageT) -> SignedMessageT:
        # def process(
        #     node: AbstractNode, msg: SignedMessageT, verify_key: VerifyKey
        # ) -> SignedMessageT:
        # TODO: Add verify_key?
        addr = msg.address

        # order is important, vm, device, domain, network
        for scope_id in [addr.vm_id, addr.device_id, addr.domain_id, addr.network_id]:
            if scope_id is not None and scope_id in node.store:
                obj = node.store[scope_id]
                try:
                    return obj.send_immediate_msg_with_reply(msg=msg)
                except Exception as e:
                    print(
                        f"{addr} in store doesnt have method send_immediate_msg_with_reply"
                    )
                    print(e)
                    pass

        try:
            in_memory_client = node.in_memory_client_registry[addr.target_id.id]
            return in_memory_client.send_immediate_msg_with_reply(msg=msg)
        except Exception as e:
            print(f"{addr} not on nodes in_memory_client. {e}")
            pass

        raise Exception(
            "Address unknown - cannot forward old_message. Throwing it away."
        )

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [ImmediateSyftMessageWithReply]
