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
from .request_api import GridRequestAPI


class GroupRequestAPI(GridRequestAPI):
    response_key = "group"

    def __init__(self, send):
        super().__init__(
            create_msg=CreateGroupMessage,
            get_msg=GetGroupMessage,
            get_all_msg=GetGroupsMessage,
            update_msg=UpdateGroupMessage,
            delete_msg=DeleteGroupMessage,
            send=send,
            response_key=GroupRequestAPI.response_key,
        )

    def __getitem__(self, key):
        return self.get(group_id=key)

    def __delitem__(self, key):
        self.delete(group_id=key)
