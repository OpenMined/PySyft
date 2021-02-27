# stdlib
from typing import Any
from typing import Dict

# third party
from pandas import DataFrame

# syft relative
from ...messages.user_messages import CreateUserMessage
from ...messages.user_messages import DeleteUserMessage
from ...messages.user_messages import GetUserMessage
from ...messages.user_messages import GetUsersMessage
from ...messages.user_messages import UpdateUserMessage
from .request_api import GridRequestAPI


class UserRequestAPI(GridRequestAPI):
    response_key = "user"

    def __init__(self, send):
        super().__init__(
            create_msg=CreateUserMessage,
            get_msg=GetUserMessage,
            get_all_msg=GetUsersMessage,
            update_msg=UpdateUserMessage,
            delete_msg=DeleteUserMessage,
            send=send,
            response_key=UserRequestAPI.response_key,
        )

    def __getitem__(self, key):
        return self.get(user_id=key)

    def __delitem__(self, key):
        self.delete(user_id=key)
