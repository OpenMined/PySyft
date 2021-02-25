# stdlib
from typing import Any
from typing import Dict

# third party
from pandas import DataFrame

# syft relative
from ...messages.group_messages import CreateGroupMessage
from ...messages.group_messages import DeleteGroupMessage
from ...messages.group_messages import GetGroupMessage
from ...messages.group_messages import GetGroupsMessage
from ...messages.group_messages import UpdateGroupMessage
from .service_request import GridServiceRequest


class GroupServiceRequest(GridServiceRequest):
    response_key = "group"

    def __init__(self, send):
        super().__init__(
            create_msg=CreateGroupMessage,
            get_msg=GetGroupMessage,
            get_all_msg=GetGroupsMessage,
            update_msg=UpdateGroupMessage,
            delete_msg=DeleteGroupMessage,
            send=send,
            response_key=GroupServiceRequest.response_key,
        )
