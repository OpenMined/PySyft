# stdlib
from typing import Callable
from typing import Dict
from typing import List
from typing import Type
from typing import Union

# third party
from nacl.signing import VerifyKey

# syft absolute
from syft.core.common.message import ImmediateSyftMessageWithReply
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.node_service.auth import service_auth
from syft.core.node.common.node_service.node_service import (
    ImmediateNodeServiceWithReply,
)
from syft.lib.python import Dict as SyftDict
from syft.lib.python import List as SyftList

# relative
from ...exceptions import AuthorizationError
from ...exceptions import MissingRequestKeyError
from ...exceptions import RequestError
from ...exceptions import RoleNotFoundError
from ...node_table.utils import model_to_json
from ..success_resp_message import SuccessResponseMessage
from .role_manager_messages import CreateRoleMessage
from .role_manager_messages import DeleteRoleMessage
from .role_manager_messages import GetRoleMessage
from .role_manager_messages import GetRoleResponse
from .role_manager_messages import GetRolesMessage
from .role_manager_messages import GetRolesResponse
from .role_manager_messages import UpdateRoleMessage

INPUT_TYPE = Union[
    Type[CreateRoleMessage],
    Type[UpdateRoleMessage],
    Type[GetRoleMessage],
    Type[GetRolesMessage],
    Type[DeleteRoleMessage],
]

INPUT_MESSAGES = Union[
    CreateRoleMessage,
    UpdateRoleMessage,
    GetRoleMessage,
    GetRolesMessage,
    DeleteRoleMessage,
]

OUTPUT_MESSAGES = Union[SuccessResponseMessage, GetRoleResponse, GetRolesResponse]


def create_role_msg(
    msg: CreateRoleMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:
    # Check key permissions
    _allowed = node.users.can_edit_roles(verify_key=verify_key)

    if not msg.name:
        raise MissingRequestKeyError(
            message="Invalid request payload, empty fields (name)!"
        )

    # Check if this role name was already registered
    try:
        node.roles.first(name=msg.name)
        raise RequestError(message="The role name already exists!")
    except RoleNotFoundError:
        pass

    if _allowed:
        node.roles.register(
            name=msg.name,
            can_triage_requests=msg.can_triage_requests,
            can_edit_settings=msg.can_edit_settings,
            can_create_users=msg.can_create_users,
            can_create_groups=msg.can_create_groups,
            can_edit_roles=msg.can_edit_roles,
            can_manage_infrastructure=msg.can_manage_infrastructure,
            can_upload_data=msg.can_upload_data,
        )
    else:
        raise AuthorizationError("You're not allowed to create a new Role!")

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="Role created successfully!",
    )


def update_role_msg(
    msg: UpdateRoleMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:

    params = {
        "name": msg.name,
        "can_triage_requests": msg.can_triage_requests,
        "can_edit_settings": msg.can_edit_settings,
        "can_create_users": msg.can_create_users,
        "can_create_groups": msg.can_create_groups,
        "can_edit_roles": msg.can_edit_roles,
        "can_manage_infrastructure": msg.can_manage_infrastructure,
        "can_upload_data": msg.can_upload_data,
    }

    if not msg.role_id:
        raise MissingRequestKeyError

    # Check Key permissions
    _allowed = node.users.can_edit_roles(verify_key=verify_key)

    if _allowed:
        node.roles.set(role_id=msg.role_id, params=params)
    else:
        raise AuthorizationError("You're not authorized to edit this role!")

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="Role updated successfully!",
    )


def get_role_msg(
    msg: GetRoleMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> GetRoleResponse:

    # Check Key permissions
    _allowed = node.users.can_triage_requests(verify_key=verify_key)

    if _allowed:
        role = node.roles.first(id=msg.role_id)
        _msg = model_to_json(role)
    else:
        raise AuthorizationError("You're not allowed to get User information!")

    return GetRoleResponse(address=msg.reply_to, content=SyftDict(_msg))


def get_all_roles_msg(
    msg: GetRolesMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> GetRolesResponse:

    _allowed = node.users.can_triage_requests(verify_key=verify_key)

    if _allowed:
        roles = node.roles.all()
        _msg = [model_to_json(role) for role in roles]
    else:
        raise AuthorizationError("You're not allowed to get Role information!")

    return GetRolesResponse(address=msg.reply_to, content=SyftList(_msg))


def del_role_msg(
    msg: DeleteRoleMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:
    _allowed = node.users.can_edit_roles(verify_key=verify_key)

    if _allowed:
        node.roles.delete(id=msg.role_id)
    else:
        raise AuthorizationError("You're not authorized to delete this role!")

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="Role has been deleted!",
    )


class RoleManagerService(ImmediateNodeServiceWithReply):
    msg_handler_map: Dict[INPUT_TYPE, Callable[..., OUTPUT_MESSAGES]] = {
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
        msg: INPUT_MESSAGES,
        verify_key: VerifyKey,
    ) -> OUTPUT_MESSAGES:
        return RoleManagerService.msg_handler_map[type(msg)](
            msg=msg, node=node, verify_key=verify_key
        )

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [
            CreateRoleMessage,
            UpdateRoleMessage,
            GetRoleMessage,
            GetRolesMessage,
            DeleteRoleMessage,
        ]
