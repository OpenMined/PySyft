# stdlib
from typing import Any

# relative
from ...abstract.node import AbstractNodeClient
from ...domain.enums import ResponseObjectEnum
from ..node_service.group_manager.group_manager_messages import CreateGroupMessage
from ..node_service.group_manager.group_manager_messages import DeleteGroupMessage
from ..node_service.group_manager.group_manager_messages import GetGroupMessage
from ..node_service.group_manager.group_manager_messages import GetGroupsMessage
from ..node_service.group_manager.group_manager_messages import UpdateGroupMessage
from .request_api import RequestAPI


class GroupRequestAPI(RequestAPI):
    def __init__(self, client: AbstractNodeClient):
        super().__init__(
            client=client,
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
