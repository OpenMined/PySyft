# stdlib
from typing import List
from typing import Optional

# third party
from nacl.signing import VerifyKey

# relative
from ......logger import debug
from ......logger import error
from ......logger import traceback_and_raise
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.message import SignedImmediateSyftMessageWithReply
from .....common.message import SignedImmediateSyftMessageWithoutReply
from .....common.message import SignedMessageT
from ....abstract.node import AbstractNode
from ...client import GET_OBJECT_TIMEOUT
from ..node_service import SignedNodeServiceWithReply
from ..node_service import SignedNodeServiceWithoutReply


class SignedMessageWithoutReplyForwardingService(SignedNodeServiceWithoutReply):
    @staticmethod
    def process(
        node: AbstractNode,
        msg: SignedImmediateSyftMessageWithoutReply,
        verify_key: Optional[VerifyKey] = None,
    ) -> Optional[SignedMessageT]:
        addr = msg.address
        debug(f"> Forwarding WithoutReply {msg.pprint} to {addr.target_emoji()}")
        # order is important, vm, device, domain, network
        for scope_id in [addr.vm_id, addr.device_id, addr.domain_id, addr.network_id]:
            if scope_id is not None and scope_id in node.store:
                obj = node.store.get(scope_id)
                func = getattr(obj, "send_immediate_msg_without_reply", None)

                if func is None:
                    error(
                        f"{addr} in store does not have method send_immediate_msg_without_reply"
                    )
                else:
                    return func(msg=msg)

        try:
            for scope_id in [
                addr.vm_id,
                addr.device_id,
                addr.domain_id,
                addr.network_id,
            ]:
                if scope_id is not None:
                    debug(f"> Lookup: {scope_id.emoji()}")
                    client = node.get_peer_client(node_id=scope_id, only_vpn=False)
                    if client:
                        return client.send_immediate_msg_without_reply(msg=msg)
                    else:
                        raise Exception
        except Exception as e:
            # TODO: Need to not catch blanket exceptions
            error(
                f"Failed to forward {msg}. "
                f"No client for {addr} in node.get_peer_client. {e}"
            )
            pass
        debug(f"> ❌ {node.pprint} 🤷🏾‍♀️ {addr.target_emoji()}")
        traceback_and_raise(
            Exception("Address unknown - cannot forward message. Throwing it away.")
        )

    @staticmethod
    def message_handler_types() -> List[type]:
        return [ImmediateSyftMessageWithoutReply]


class SignedMessageWithReplyForwardingService(SignedNodeServiceWithReply):
    @staticmethod
    def process(
        node: AbstractNode,
        msg: SignedImmediateSyftMessageWithReply,
        verify_key: Optional[VerifyKey] = None,
    ) -> SignedImmediateSyftMessageWithoutReply:
        addr = msg.address
        debug(f"> Forwarding WithReply {msg.pprint} to {addr.target_emoji()}")

        # order is important, vm, device, domain, network
        for scope_id in [addr.vm_id, addr.device_id, addr.domain_id, addr.network_id]:
            if scope_id is not None and scope_id in node.store:
                obj = node.store.get(scope_id)
                func = getattr(obj, "send_immediate_msg_with_reply", None)
                if func is None or not callable(func):
                    error(
                        f"{addr} in store does not have method send_immediate_msg_with_reply"
                    )
                else:
                    return func(msg=msg, timeout=15)

        try:
            for scope_id in [
                addr.vm_id,
                addr.device_id,
                addr.domain_id,
                addr.network_id,
            ]:
                if scope_id is not None:
                    debug(f"> Lookup: {scope_id.emoji()}")
                    client = node.get_peer_client(node_id=scope_id, only_vpn=False)
                    if client:
                        return client.send_immediate_msg_with_reply(msg=msg, timeout=GET_OBJECT_TIMEOUT)  # type: ignore
                    else:
                        raise Exception
        except Exception as e:
            # TODO: Need to not catch blanket exceptions
            error(
                f"Failed to forward {msg}. "
                f"No client for {addr} in node.get_peer_client. {e}"
            )
            pass
        debug(f"> ❌ {node.pprint} 🤷🏾‍♀️ {addr.target_emoji()}")
        traceback_and_raise(
            Exception("Address unknown - cannot forward message. Throwing it away.")
        )

    @staticmethod
    def message_handler_types() -> List[type]:
        return [ImmediateSyftMessageWithReply]
