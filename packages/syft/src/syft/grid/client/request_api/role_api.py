# stdlib
from typing import Any
from typing import Callable

# syft relative
from ...messages.role_messages import CreateRoleMessage
from ...messages.role_messages import DeleteRoleMessage
from ...messages.role_messages import GetRoleMessage
from ...messages.role_messages import GetRolesMessage
from ...messages.role_messages import UpdateRoleMessage
from ..enums import ResponseObjectEnum
from .request_api import GridRequestAPI


class RoleRequestAPI(GridRequestAPI):
    def __init__(self, send: Callable):
        super().__init__(
            create_msg=CreateRoleMessage,
            get_msg=GetRoleMessage,
            get_all_msg=GetRolesMessage,
            update_msg=UpdateRoleMessage,
            delete_msg=DeleteRoleMessage,
            send=send,
            response_key=ResponseObjectEnum.ROLE,
        )

    def __getitem__(self, key: int) -> Any:
        return self.get(role_id=key)

    def __delitem__(self, key: int) -> None:
        self.delete(role_id=key)
