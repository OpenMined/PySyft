# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from nacl.signing import VerifyKey

# relative
from ....abstract.node_service_interface import NodeServiceInterface
from ..auth import service_auth
from ..node_service import ImmediateNodeServiceWithReply
from .user_auth_messages import UserLoginMessage
from .user_auth_messages import UserLoginMessageWithReply
from .user_auth_messages import UserLoginReplyMessage


class UserLoginService(ImmediateNodeServiceWithReply):
    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: NodeServiceInterface,
        msg: UserLoginMessage,
        verify_key: Optional[VerifyKey] = None,
    ) -> UserLoginReplyMessage:
        # this service requires no verify_key because its currently public
        result = msg.payload.run(node=node, verify_key=verify_key)
        return UserLoginMessageWithReply(kwargs=result).back_to(address=msg.reply_to)

    @staticmethod
    def message_handler_types() -> List[Type[UserLoginMessage]]:
        return [UserLoginMessage]
