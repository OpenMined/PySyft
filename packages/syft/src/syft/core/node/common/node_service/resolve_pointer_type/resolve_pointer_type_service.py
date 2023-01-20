"""This file defines the class to process the request to regenerate
the type pointer corresponding to the current pointer."""


# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from nacl.signing import VerifyKey

# relative
from ....abstract.node import AbstractNode
from ..node_service import ImmediateNodeServiceWithReply
from .resolve_pointer_type_messages import ResolvePointerTypeAnswerMessage
from .resolve_pointer_type_messages import ResolvePointerTypeMessage


class ResolvePointerTypeService(ImmediateNodeServiceWithReply):
    """Validate the current pointer and return the corresponding type resolved pointer."""

    @staticmethod
    def process(
        node: AbstractNode,
        msg: ResolvePointerTypeMessage,
        verify_key: Optional[VerifyKey] = None,
    ) -> ResolvePointerTypeAnswerMessage:
        """Return the type pointer corresponding to the current pointer.

        Args:
            node (AbstractNode): node client.
            msg (ResolvePointerTypeMessage): address of the pointer whose type needs to be resolved.
            verify_key (Optional[VerifyKey], optional): digital signature of the user. Defaults to None.

        Returns:
            ResolvePointerTypeAnswerMessage: details of the generated pointer type.
        """
        # TODO: refactor so we can get the pointer without deserializing the whole
        # object which could be over in the blob store
        object = node.store.get(msg.id_at_location, proxy_only=False)
        type_qualname = object.object_qualname
        return ResolvePointerTypeAnswerMessage(
            address=msg.reply_to, type_path=type_qualname
        )

    @staticmethod
    def message_handler_types() -> List[Type[ResolvePointerTypeMessage]]:
        return [ResolvePointerTypeMessage]
