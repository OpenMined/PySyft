# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from nacl.signing import VerifyKey

# relative
from ......util import traceback_and_raise
from ....abstract.node_service_interface import NodeServiceInterface
from ..auth import service_auth
from ..node_service import ImmediateNodeServiceWithReply
from .sleep_messages import SleepMessage
from .sleep_messages import SleepMessageWithReply
from .sleep_messages import SleepReplyMessage


class SleepService(ImmediateNodeServiceWithReply):

    # TODO: Change this to not be available for guests
    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: NodeServiceInterface,
        msg: SleepMessage,
        verify_key: Optional[VerifyKey] = None,
    ) -> SleepReplyMessage:
        if verify_key is None:
            traceback_and_raise("Can't process PingService with no verification key.")

        result = msg.payload.run(node=node, verify_key=verify_key)
        return SleepMessageWithReply(kwargs=result).back_to(address=msg.reply_to)

    @staticmethod
    def message_handler_types() -> List[Type[SleepMessage]]:
        return [SleepMessage]
