# stdlib
from typing import Any

# syft absolute
from syft.core.node.abstract.node import AbstractNodeClient
from syft.core.node.common.node_service.role_manager.role_manager_messages import (
    CreateRoleMessage,
)
from syft.core.node.common.node_service.role_manager.role_manager_messages import (
    DeleteRoleMessage,
)
from syft.core.node.common.node_service.role_manager.role_manager_messages import (
    GetRoleMessage,
)
from syft.core.node.common.node_service.role_manager.role_manager_messages import (
    GetRolesMessage,
)
from syft.core.node.common.node_service.role_manager.role_manager_messages import (
    UpdateRoleMessage,
)

# relative
from ....node.domain.enums import ResponseObjectEnum
from ...common.client_manager.request_api import RequestAPI


class RoleRequestAPI(RequestAPI):
    def __init__(self, client: AbstractNodeClient):
        super().__init__(
            client=client,
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
