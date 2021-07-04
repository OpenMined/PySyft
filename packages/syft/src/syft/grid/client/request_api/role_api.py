# stdlib
from typing import Any
from typing import Callable
from typing import Type

# relative
from ....core.node.common.node import Node
from ....core.node.domain.enums import ResponseObjectEnum
from ...messages.role_messages import CreateRoleMessage
from ...messages.role_messages import DeleteRoleMessage
from ...messages.role_messages import GetRoleMessage
from ...messages.role_messages import GetRolesMessage
from ...messages.role_messages import UpdateRoleMessage
from .request_api import GridRequestAPI


class RoleRequestAPI(GridRequestAPI):
    def __init__(self, node: Type[Node]):
        super().__init__(
            node=node,
            create_msg=CreateRoleMessage,
            get_msg=GetRoleMessage,
            get_all_msg=GetRolesMessage,
            update_msg=UpdateRoleMessage,
            delete_msg=DeleteRoleMessage,
            response_key=ResponseObjectEnum.ROLE,
        )

    def __getitem__(self, key: int) -> Any:
        return self.get(role_id=key)

    def __delitem__(self, key: int) -> None:
        self.delete(role_id=key)
