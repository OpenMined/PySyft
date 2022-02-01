# stdlib
from typing import Optional

# third party
from nacl.signing import VerifyKey

# relative
from ....core.node.abstract.node_service_interface import NodeServiceInterface
from ....core.node.common.node_service.auth import service_auth
from ....core.node.common.node_service.generic_payload.messages import (
    GenericPayloadMessage,
)
from ....core.node.common.node_service.generic_payload.messages import (
    GenericPayloadMessageWithReply,
)
from ....core.node.common.node_service.node_service import NodeService
from .registry import DomainMessageRegistry


class DomainServiceClass(NodeService):
    @staticmethod
    @service_auth(guests_welcome=True)  # Service level authentication
    def process(
        node: NodeServiceInterface,
        msg: GenericPayloadMessage,
        verify_key: Optional[VerifyKey] = None,
    ) -> GenericPayloadMessageWithReply:

        result = msg.payload.run(node=node, verify_key=verify_key)
        payload_class = msg.payload.__class__
        return payload_class(kwargs=result).back_to(address=msg.reply_to)

    @staticmethod
    def message_handler_types() -> list:
        registered_messages = DomainMessageRegistry().get_registered_messages()
        return registered_messages
