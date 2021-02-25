# syft relative
from ...messages.role_messages import CreateRoleMessage
from ...messages.role_messages import DeleteRoleMessage
from ...messages.role_messages import GetRoleMessage
from ...messages.role_messages import GetRolesMessage
from ...messages.role_messages import UpdateRoleMessage
from .service_request import GridServiceRequest


class RoleServiceRequest(GridServiceRequest):
    response_key = "role"

    def __init__(self, send):
        super().__init__(
            create_msg=CreateRoleMessage,
            get_msg=GetRoleMessage,
            get_all_msg=GetRolesMessage,
            update_msg=UpdateRoleMessage,
            delete_msg=DeleteRoleMessage,
            send=send,
            response_key=RoleServiceRequest.response_key,
        )
