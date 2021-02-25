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
from .service_request import GridServiceRequest


class UserServiceRequest(GridServiceRequest):
    response_key = "user"

    def __init__(self, send):
        super().__init__(
            create_msg=CreateUserMessage,
            get_msg=GetUserMessage,
            get_all_msg=GetUsersMessage,
            update_msg=UpdateUserMessage,
            delete_msg=DeleteUserMessage,
            send=send,
            response_key=UserServiceRequest.response_key,
        )
