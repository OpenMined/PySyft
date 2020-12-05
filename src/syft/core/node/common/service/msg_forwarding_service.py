# stdlib
from typing import List
from typing import Optional

# third party
from loguru import logger

# syft relative
from .....decorators import syft_decorator
from ....common.message import ImmediateSyftMessageWithReply
from ....common.message import ImmediateSyftMessageWithoutReply
from ....common.message import SignedImmediateSyftMessageWithReply
from ....common.message import SignedImmediateSyftMessageWithoutReply
from ....common.message import SignedMessageT
from ...abstract.node import AbstractNode
from .node_service import SignedNodeServiceWithReply
from .node_service import SignedNodeServiceWithoutReply


class SignedMessageWithoutReplyForwardingService(SignedNodeServiceWithoutReply):
    @staticmethod
    @syft_decorator(typechecking=True)
    def process(
        node: AbstractNode, msg: SignedImmediateSyftMessageWithoutReply
    ) -> Optional[SignedMessageT]:
        addr = msg.address
        logger.debug(f"> Forwarding WithoutReply {msg.pprint} to {addr.target_emoji()}")
        # order is important, vm, device, domain, network
        for scope_id in [addr.vm_id, addr.device_id, addr.domain_id, addr.network_id]:
            if scope_id is not None and scope_id in node.store:
                obj = node.store[scope_id]
                try:
                    return obj.send_immediate_msg_without_reply(msg=msg)
                except Exception as e:
                    # TODO: Need to not catch blanket exceptions
                    print(
                        f"{addr} in store does not have method send_immediate_msg_without_reply"
                    )
                    print(e)
                    pass

        try:
            for scope_id in [
                addr.vm_id,
                addr.device_id,
                addr.domain_id,
                addr.network_id,
            ]:
                if scope_id is not None:
                    logger.debug(f"> Lookup: {scope_id.emoji()}")
                    if scope_id in node.in_memory_client_registry:
                        in_memory_client = node.in_memory_client_registry[scope_id]
                        return in_memory_client.send_immediate_msg_without_reply(
                            msg=msg
                        )
        except Exception as e:
            # TODO: Need to not catch blanket exceptions
            print(f"{addr} not on nodes in_memory_client. {e}")
            pass
        logger.debug(f"> ❌ {node.pprint} 🤷🏾‍♀️ {addr.target_emoji()}")
        raise Exception("Address unknown - cannot forward message. Throwing it away.")

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [ImmediateSyftMessageWithoutReply]


class SignedMessageWithReplyForwardingService(SignedNodeServiceWithReply):
    @staticmethod
    @syft_decorator(typechecking=True)
    def process(
        node: AbstractNode, msg: SignedImmediateSyftMessageWithReply
    ) -> SignedImmediateSyftMessageWithoutReply:
        # def process(
        #     node: AbstractNode, msg: SignedMessageT, verify_key: VerifyKey
        # ) -> SignedMessageT:
        # TODO: Add verify_key?
        addr = msg.address
        logger.debug(f"> Forwarding WithReply {msg.pprint} to {addr.target_emoji()}")

        # order is important, vm, device, domain, network
        for scope_id in [addr.vm_id, addr.device_id, addr.domain_id, addr.network_id]:
            if scope_id is not None and scope_id in node.store:
                obj = node.store[scope_id]
                try:
                    return obj.send_immediate_msg_with_reply(msg=msg)
                except Exception as e:
                    # TODO: Need to not catch blanket exceptions
                    print(
                        f"{addr} in store does not have method send_immediate_msg_with_reply"
                    )
                    print(e)
                    pass

        try:
            for scope_id in [
                addr.vm_id,
                addr.device_id,
                addr.domain_id,
                addr.network_id,
            ]:
                if scope_id is not None:
                    logger.debug(f"> Lookup: {scope_id.emoji()}")
                    if scope_id in node.in_memory_client_registry:
                        in_memory_client = node.in_memory_client_registry[scope_id]
                        return in_memory_client.send_immediate_msg_without_reply(
                            msg=msg
                        )
        except Exception as e:
            # TODO: Need to not catch blanket exceptions
            print(f"{addr} not on nodes in_memory_client. {e}")
            pass
        logger.debug(f"> ❌ {node.pprint} 🤷🏾‍♀️ {addr.target_emoji()}")
        raise Exception("Address unknown - cannot forward message. Throwing it away.")

    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [ImmediateSyftMessageWithReply]
