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
from .peer_discovery_messages import PeerDiscoveryMessage
from .peer_discovery_messages import PeerDiscoveryMessageWithReply
from .peer_discovery_messages import PeerDiscoveryReplyMessage


class PeerDiscoveryService(ImmediateNodeServiceWithReply):
    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode,
        msg: PeerDiscoveryMessage,
        verify_key: Optional[VerifyKey] = None,
    ) -> PeerDiscoveryReplyMessage:
        # this service requires no verify_key because its currently public
        result = msg.payload.run(node=node, verify_key=verify_key)
        return PeerDiscoveryMessageWithReply(kwargs=result).back_to(
            address=msg.reply_to
        )

    @staticmethod
    def message_handler_types() -> List[Type[PeerDiscoveryMessage]]:
        return [PeerDiscoveryMessage]
