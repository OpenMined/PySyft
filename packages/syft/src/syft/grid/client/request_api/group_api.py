# stdlib
from typing import Any
from typing import Type

# relative
from ....core.node.common.node import Node
from ....core.node.domain.enums import ResponseObjectEnum
from ...messages.group_messages import CreateGroupMessage
from ...messages.group_messages import DeleteGroupMessage
from ...messages.group_messages import GetGroupMessage
from ...messages.group_messages import GetGroupsMessage
from ...messages.group_messages import UpdateGroupMessage
from .request_api import GridRequestAPI


class GroupRequestAPI(GridRequestAPI):
    def __init__(self, node: Type[Node]):
        super().__init__(
            node=node,
            create_msg=CreateGroupMessage,
            get_msg=GetGroupMessage,
            get_all_msg=GetGroupsMessage,
            update_msg=UpdateGroupMessage,
            delete_msg=DeleteGroupMessage,
            response_key=ResponseObjectEnum.GROUP,
        )

    def __getitem__(self, key: int) -> Any:
        return self.get(group_id=key)

    def __delitem__(self, key: int) -> Any:
        self.delete(group_id=key)
