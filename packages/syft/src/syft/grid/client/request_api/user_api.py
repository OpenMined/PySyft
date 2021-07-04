# stdlib
from typing import Any
from typing import Type

# relative
from ....core.node.common.node import Node
from ....core.node.domain.enums import ResponseObjectEnum
from ...messages.user_messages import CreateUserMessage
from ...messages.user_messages import DeleteUserMessage
from ...messages.user_messages import GetUserMessage
from ...messages.user_messages import GetUsersMessage
from ...messages.user_messages import UpdateUserMessage
from .request_api import GridRequestAPI


class UserRequestAPI(GridRequestAPI):
    def __init__(self, node: Type[Node]):
        super().__init__(
            node=node,
            create_msg=CreateUserMessage,
            get_msg=GetUserMessage,
            get_all_msg=GetUsersMessage,
            update_msg=UpdateUserMessage,
            delete_msg=DeleteUserMessage,
            response_key=ResponseObjectEnum.USER,
        )

    def __getitem__(self, key: int) -> Any:
        return self.get(user_id=key)

    def __delitem__(self, key: int) -> None:
        self.delete(user_id=key)
