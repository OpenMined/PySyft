# syft relative
from ...messages.role_messages import CreateRoleMessage
from ...messages.role_messages import DeleteRoleMessage
from ...messages.role_messages import GetRoleMessage
from ...messages.role_messages import GetRolesMessage
from ...messages.role_messages import UpdateRoleMessage
from .request_api import GridRequestAPI


class RoleRequestAPI(GridRequestAPI):
    response_key = "role"

    def __init__(self, send):
        super().__init__(
            create_msg=CreateRoleMessage,
            get_msg=GetRoleMessage,
            get_all_msg=GetRolesMessage,
            update_msg=UpdateRoleMessage,
            delete_msg=DeleteRoleMessage,
            send=send,
            response_key=RoleRequestAPI.response_key,
        )

    def __getitem__(self, key):
        return self.get(role_id=key)

    def __delitem__(self, key):
        self.delete(role_id=key)
