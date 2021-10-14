# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from nacl.signing import VerifyKey

# relative
from ......util import traceback_and_raise
from ....abstract.node import AbstractNode
from ..auth import service_auth
from ..node_service import ImmediateNodeServiceWithReply
from .vpn_messages import VPNConnectMessage
from .vpn_messages import VPNConnectMessageWithReply
from .vpn_messages import VPNConnectReplyMessage


class VPNConnectService(ImmediateNodeServiceWithReply):
    @staticmethod
    # @service_auth(root_only=True)
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode,
        msg: VPNConnectMessage,
        verify_key: Optional[VerifyKey] = None,
    ) -> VPNConnectReplyMessage:
        if verify_key is None:
            traceback_and_raise(
                f"Can't process VPNConnectService with no verification key."
            )

        result = msg.payload.run(node=node, verify_key=verify_key)
        return VPNConnectMessageWithReply(kwargs=result).back_to(address=msg.reply_to)

    @staticmethod
    def message_handler_types() -> List[Type[VPNConnectMessage]]:
        return [VPNConnectMessage]
