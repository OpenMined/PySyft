# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from nacl.signing import VerifyKey

# relative
from ....abstract.node_service_interface import NodeServiceInterface
from ..auth import service_auth
from ..node_service import NodeService
from ..registry import DomainServiceRegistry
from ..registry import NetworkServiceRegistry
from ..user_manager.new_user_messages import GetMessage
from ..user_manager.new_user_messages import GetUserMessage


class UserService(NetworkServiceRegistry, DomainServiceRegistry, NodeService):
    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: NodeServiceInterface,
        msg: GetMessage,
        verify_key: Optional[VerifyKey] = None,
    ) -> GetUserMessage:

        result = msg.payload.run(node=node, verify_key=verify_key)
        return GetUserMessage(kwargs=result).back_to(address=msg.reply_to)

    @staticmethod
    def message_handler_types() -> List[Type[GetMessage]]:
        return [GetMessage]
