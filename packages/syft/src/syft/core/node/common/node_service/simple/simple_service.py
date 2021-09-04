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
from .simple_messages import SimpleMessage
from .simple_messages import SimpleReplyMessage


class SimpleService(ImmediateNodeServiceWithReply):
    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode,
        msg: SimpleMessage,
        verify_key: Optional[VerifyKey] = None,
    ) -> SimpleReplyMessage:
        if verify_key is None:
            traceback_and_raise(
                "Can't process an GetReprService with no verification key."
            )

        result = msg.payload.run(node=node, verify_key=verify_key)

        return SimpleReplyMessage(payload=result, address=msg.reply_to)

    @staticmethod
    def message_handler_types() -> List[Type[SimpleMessage]]:
        return [SimpleMessage]
