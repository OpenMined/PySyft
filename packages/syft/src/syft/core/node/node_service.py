# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from nacl.signing import VerifyKey

# relative
from ...core.node.abstract.node_service_interface import NodeServiceInterface
from ...core.node.common.node_service.auth import service_auth
from ...core.node.common.node_service.generic_payload.syft_message import (
    NewSyftMessage as SyftMessage,
)
from ...core.node.common.node_service.node_service import NodeService
from .abstract_node_msg_registry import AbstractNodeMessageRegistry


class NodeServiceClass(NodeService):
    registry_type: Type = AbstractNodeMessageRegistry

    @staticmethod
    @service_auth(guests_welcome=True)  # Service level authentication
    def process(
        node: NodeServiceInterface,
        msg: SyftMessage,
        verify_key: Optional[VerifyKey] = None,
    ) -> SyftMessage:
        """A single service to execute messages for the domain client.

        Args:
            node (NodeServiceInterface): domain service interface.
            msg (NewSyftMessage): the message that needs to executed.
            verify_key (Optional[VerifyKey], optional): unique verification of the user. Defaults to None.

        Returns:
            SyftMessage: response message.

        Note:
            TODO: Error and Exception Handling. Errors can be messages of `NewSyftMessage` type.
            Instead of holding the result, they will hold the errors. In short an error is a `NewSyftMessage`
            itself and is propagated to the end user. While when an exception is occurred, the flow should
            be broken to prevent any further execution and an exception type message should be raised.
            The error/exception message handles if it can be private or public.
        """
        msg.check_permissions(node=node, verify_key=verify_key)
        result = msg.run(node=node, verify_key=verify_key).dict()  # type: ignore
        payload_class = msg.__class__
        return payload_class(address=msg.reply_to, kwargs=result, reply=True)  # type: ignore

    @classmethod
    def message_handler_types(cls) -> List:
        return cls.registry_type.get_registered_messages()
