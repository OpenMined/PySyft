"""This file defines all the functions/classes to perform any CRUD operation on
the Role table, for a given domain node, in a RESTful manner.
"""

# stdlib
from typing import Callable
from typing import Dict
from typing import List
from typing import Type
from typing import Union

# third party
from nacl.signing import VerifyKey

# relative
from .....common.message import ImmediateSyftMessageWithReply
from ....domain_interface import DomainInterface
from ...exceptions import AuthorizationError
from ...exceptions import MissingRequestKeyError
from ...exceptions import RequestError
from ...exceptions import RoleNotFoundError
from ...node_table.utils import model_to_json
from ..auth import service_auth
from ..node_service import ImmediateNodeServiceWithReply
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
    node: DomainInterface,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:
    """Creates a new role in the database.

    Args:
        msg (CreateRoleMessage): details of the role.
        node (DomainInterface): domain node.
        verify_key (VerifyKey): public digital signature/key of the user.

    Raises:
        MissingRequestKeyError: If name of the role is missing.
        RequestError: If role already exists.
        AuthorizationError: If user does not have permissions to create new role.

    Returns:
        SuccessResponseMessage: Success message on role creation.
    """
    # Check if user has permissions to create new roles
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
            can_make_data_requests=msg.can_make_data_requests,
            can_triage_data_requests=msg.can_triage_data_requests,
            can_manage_privacy_budget=msg.can_manage_privacy_budget,
            can_create_users=msg.can_create_users,
            can_manage_users=msg.can_manage_users,
            can_edit_roles=msg.can_edit_roles,
            can_manage_infrastructure=msg.can_manage_infrastructure,
            can_upload_data=msg.can_upload_data,
            can_upload_legal_document=msg.can_upload_legal_document,
            can_edit_domain_settings=msg.can_edit_domain_settings,
        )
    else:
        raise AuthorizationError("You're not allowed to create a new Role!")

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="Role created successfully!",
    )


def update_role_msg(
    msg: UpdateRoleMessage,
    node: DomainInterface,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:
    """Updates the properties of the given role.

    Args:
        msg (UpdateRoleMessage): stores msg address and properties of the role to be updated.
        node (DomainInterface): domain node.
        verify_key (VerifyKey): public digital signature of the user.

    Raises:
        MissingRequestKeyError: If the role id does not exist in the `msg`.
        AuthorizationError: If user does not have permissions to perform the update operation.

    Returns:
        SuccessResponseMessage: Message on successfully updating the role.
    """

    params = {
        "name": msg.name,
        "can_make_data_requests": msg.can_make_data_requests,
        "can_triage_data_requests": msg.can_triage_data_requests,
        "can_manage_privacy_budget": msg.can_manage_privacy_budget,
        "can_create_users": msg.can_create_users,
        "can_manage_users": msg.can_manage_users,
        "can_edit_roles": msg.can_edit_roles,
        "can_manage_infrastructure": msg.can_manage_infrastructure,
        "can_upload_data": msg.can_upload_data,
        "can_upload_legal_document": msg.can_upload_legal_document,
        "can_edit_domain_settings": msg.can_edit_domain_settings,
    }

    if not msg.role_id:
        raise MissingRequestKeyError

    # Check if user has permissions to edit roles
    _allowed = node.users.can_edit_roles(verify_key=verify_key) and str(
        msg.role_id
    ) != str(node.roles.owner_role.id)

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
    node: DomainInterface,
    verify_key: VerifyKey,
) -> GetRoleResponse:
    """Retrieves details of a role.

    Args:
        msg (GetRoleMessage): stores msg address and role id.
        node (DomainInterface): domain node.
        verify_key (VerifyKey): public digital signature of the user.

    Raises:
        AuthorizationError: If user does not have permissions to get role information.

    Returns:
        GetRoleResponse: details of the role.
    """

    # Check if user has permissions to triage requests
    _allowed = node.users.can_triage_requests(verify_key=verify_key)

    if _allowed:
        role = node.roles.first(id=msg.role_id)
        _msg = model_to_json(role)
    else:
        raise AuthorizationError(
            "get_role_msg You're not allowed to get User information!"
        )

    return GetRoleResponse(address=msg.reply_to, content=_msg)


def get_all_roles_msg(
    msg: GetRolesMessage,
    node: DomainInterface,
    verify_key: VerifyKey,
) -> GetRolesResponse:
    """Retrieves details of the all available roles.

    Args:
        msg (GetRolesMessage): stores the address of the message.
        node (DomainInterface): domain node.
        verify_key (VerifyKey): public digital signature of the user.

    Raises:
        AuthorizationError: If user does not have permissions to access roles.

    Returns:
        GetRolesResponse: stores the details for all the roles as a list.
    """

    # Check if user has permissions to view roles
    _allowed = node.users.can_triage_requests(verify_key=verify_key)

    if _allowed:
        roles = node.roles.all()

        _msg = [model_to_json(role) for role in roles if role.name != "Owner"]
    else:
        raise AuthorizationError("You're not allowed to get Role information!")

    return GetRolesResponse(address=msg.reply_to, content=_msg)


def del_role_msg(
    msg: DeleteRoleMessage,
    node: DomainInterface,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:
    """Delete the role corresponding to the given role id.

    Args:
        msg (DeleteRoleMessage): stores the msg address and id of the role to be deleted.
        node (DomainInterface): domain node.
        verify_key (VerifyKey): public digital signature of the user.

    Raises:
        AuthorizationError: If user does not have permissions to edit roles.

    Returns:
        SuccessResponseMessage: stores the response msg on successful role deletion.
    """

    # Check if user has permissions to edit roles
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
    """A class to handle all operations performed on the Role table."""

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
        node: DomainInterface,
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
