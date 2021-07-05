# stdlib
from typing import Any
from typing import Type

# relative
from ....node.common.node import Node
from ....node.domain.enums import ResponseObjectEnum
from ...common.request_api.request_api import RequestAPI
from ..service.group_manager.group_manager_messages import CreateGroupMessage
from ..service.group_manager.group_manager_messages import DeleteGroupMessage
from ..service.group_manager.group_manager_messages import GetGroupMessage
from ..service.group_manager.group_manager_messages import GetGroupsMessage
from ..service.group_manager.group_manager_messages import UpdateGroupMessage


class GroupRequestAPI(RequestAPI):
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
