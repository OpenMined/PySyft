# stdlib
import secrets
from typing import List
from typing import Type
from typing import Union

# third party
from nacl.signing import VerifyKey

# syft relative
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.service.auth import service_auth
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithReply
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithoutReply
from syft.core.common.message import ImmediateSyftMessageWithReply

from syft.grid.messages.role_messages import (
    CreateRoleMessage,
    CreateRoleResponse,
    GetRoleMessage,
    GetRoleResponse,
    UpdateRoleMessage,
    UpdateRoleResponse,
    DeleteRoleMessage,
    DeleteRoleResponse,
    GetRolesMessage,
    GetRolesResponse,
)


def create_role_msg(
    msg: CreateRoleMessage,
) -> CreateRoleResponse:
    return CreateRoleResponse(
        address=msg.reply_to,
        status_code=200,
        content={"msg": "Role created successfully!"},
    )


def update_role_msg(
    msg: UpdateRoleMessage,
) -> UpdateRoleResponse:
    return UpdateRoleResponse(
        address=msg.reply_to,
        status_code=200,
        content={"msg": "Role updated successfully!"},
    )


def get_role_msg(
    msg: GetRoleMessage,
) -> GetRoleResponse:
    return GetRoleResponse(
        address=msg.reply_to,
        status_code=200,
        content={
            "name": "mario mario",
            "can_triage_requests": False,
            "can_edit_settings": False,
            "can_create_users": True,
            "can_create_groups": True,
            "can_edit_roles": False,
            "can_manage_infrastructure": False,
            "can_upload_data": False,
        },
    )


def get_all_roles_msg(
    msg: GetRolesMessage,
) -> GetRolesResponse:
    return GetRolesResponse(
        address=msg.reply_to,
        status_code=200,
        content={
            "roles": {
                "asd64f85as": {
                    "name": "role name1",
                    "can_triage_requests": False,
                    "can_edit_settings": False,
                    "can_create_users": True,
                    "can_create_groups": True,
                    "can_edit_roles": False,
                    "can_manage_infrastructure": False,
                    "can_upload_data": False,
                },
                "esad556d1a": {
                    "name": "role name2",
                    "can_triage_requests": False,
                    "can_edit_settings": False,
                    "can_create_users": True,
                    "can_create_groups": True,
                    "can_edit_roles": False,
                    "can_manage_infrastructure": False,
                    "can_upload_data": False,
                },
            }
        },
    )


def del_role_msg(
    msg: DeleteRoleMessage,
) -> DeleteRoleResponse:
    print("I'm here!")
    return DeleteRoleResponse(
        address=msg.reply_to,
        status_code=200,
        content={"msg": "Role has been deleted!"},
    )


class RoleManagerService(ImmediateNodeServiceWithReply):

    msg_handler_map = {
        CreateRoleMessage: create_role_msg,
        UpdateRoleMessage: update_role_msg,
        GetRoleMessage: get_role_msg,
        GetRolesMessage: get_all_roles_msg,
        DeleteRoleMessage: del_role_msg,
    }

    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode,
        msg: Union[
            CreateRoleMessage,
            UpdateRoleMessage,
            GetRoleMessage,
            GetRolesMessage,
            DeleteRoleMessage,
        ],
        verify_key: VerifyKey,
    ) -> Union[
        CreateRoleResponse,
        UpdateRoleResponse,
        GetRoleResponse,
        GetRolesResponse,
        DeleteRoleResponse,
    ]:
        return RoleManagerService.msg_handler_map[type(msg)](msg=msg)

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [
            CreateRoleMessage,
            UpdateRoleMessage,
            GetRoleMessage,
            GetRolesMessage,
            DeleteRoleMessage,
        ]
