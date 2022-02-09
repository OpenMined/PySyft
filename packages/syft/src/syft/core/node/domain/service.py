# stdlib
from typing import Optional

# third party
from nacl.signing import VerifyKey

# relative
from ....core.node.abstract.node_service_interface import NodeServiceInterface
from ....core.node.common.node_service.auth import service_auth
from ....core.node.common.node_service.generic_payload.syft_message import SyftMessage
from ....core.node.common.node_service.node_service import NodeService
from .registry import DomainMessageRegistry


class DomainServiceClass(NodeService):
    @staticmethod
    @service_auth(guests_welcome=True)  # Service level authentication
    def process(
        node: NodeServiceInterface,
        msg: SyftMessage,
        verify_key: Optional[VerifyKey] = None,
    ) -> SyftMessage:
        result = msg.run(node=node, verify_key=verify_key).dict()  # type: ignore
        payload_class = msg.__class__
        return payload_class(address=msg.reply_to, kwargs=result, reply=True)  # type: ignore

    @staticmethod
    def message_handler_types() -> list:
        registered_messages = DomainMessageRegistry().get_registered_messages()
        return registered_messages
