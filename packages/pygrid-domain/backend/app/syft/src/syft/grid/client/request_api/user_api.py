# stdlib
from typing import Any
from typing import Callable

# syft relative
from ...messages.user_messages import CreateUserMessage
from ...messages.user_messages import DeleteUserMessage
from ...messages.user_messages import GetUserMessage
from ...messages.user_messages import GetUsersMessage
from ...messages.user_messages import UpdateUserMessage
from ..enums import ResponseObjectEnum
from .request_api import GridRequestAPI


class UserRequestAPI(GridRequestAPI):
    def __init__(self, send: Callable):
        super().__init__(
            create_msg=CreateUserMessage,
            get_msg=GetUserMessage,
            get_all_msg=GetUsersMessage,
            update_msg=UpdateUserMessage,
            delete_msg=DeleteUserMessage,
            send=send,
            response_key=ResponseObjectEnum.USER,
        )

    def __getitem__(self, key: int) -> Any:
        return self.get(user_id=key)

    def __delitem__(self, key: int) -> None:
        self.delete(user_id=key)
