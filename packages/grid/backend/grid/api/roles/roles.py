# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Union

# third party
from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from fastapi.responses import JSONResponse
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey

# syft absolute
from syft.core.node.common.action.exception_action import ExceptionMessage

# syft
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

# grid absolute
from grid.api.dependencies.current_user import get_current_user
from grid.core.node import node

router = APIRouter()


@router.post("", status_code=201, response_class=JSONResponse)
def create_role_route(
    current_user: Any = Depends(get_current_user),
    name: str = Body(False, example="Researcher"),
    can_make_data_requests: bool = Body(False, example="false"),
    can_triage_data_requests: bool = Body(False, example="false"),
    can_manage_privacy_budget: bool = Body(False, example="false"),
    can_create_users: bool = Body(False, example="false"),
    can_manage_users: bool = Body(False, example="false"),
    can_edit_roles: bool = Body(False, example="false"),
    can_manage_infrastructure: bool = Body(False, example="false"),
    can_upload_data: bool = Body(False, example="false"),
    can_upload_legal_document: bool = Body(False, example="false"),
    can_edit_domain_settings: bool = Body(False, example="false"),
) -> Dict[str, str]:
    """Creates a new PyGrid role.

    Args:
        current_user : Current session.
        name: Role name.
        can_triage_requests: Allow role to triage requests.
        can_edit_settings: Allow role to edit settings.
        can_create_users: Allow role to create users.
        can_create_groups: Allow role to create groups.
        can_edit_roles: Allow role to edit other roles.
        can_manage_infrastructure: Allow role to manage Node's infrastructure.
        can_upload_data: Allow role to upload data.
    Returns:
        resp: JSON structure containing a log message.
    """
    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = CreateRoleMessage(
        address=node.address,
        name=name,
        can_make_data_requests=can_make_data_requests,
        can_triage_data_requests=can_triage_data_requests,
        can_manage_privacy_budget=can_manage_privacy_budget,
        can_create_users=can_create_users,
        can_manage_users=can_manage_users,
        can_edit_roles=can_edit_roles,
        can_manage_infrastructure=can_manage_infrastructure,
        can_upload_data=can_upload_data,
        can_upload_legal_document=can_upload_legal_document,
        can_edit_domain_settings=can_edit_domain_settings,
        reply_to=node.address,
    ).sign(signing_key=user_key)

    # Process syft message
    reply = node.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    if isinstance(reply, ExceptionMessage):
        return {"error": reply.exception_msg}
    else:
        return {"message": reply.resp_msg}


@router.get("", status_code=200, response_class=JSONResponse)
def get_all_roles_route(
    current_user: Any = Depends(get_current_user),
) -> Union[Dict[str, str], List[Dict[str, Any]]]:
    """Retrieves all registered roles

    Args:
        current_user : Current session.
    Returns:
        resp: JSON structure containing registered roles.
    """
    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = GetRolesMessage(address=node.address, reply_to=node.address).sign(
        signing_key=user_key
    )

    # Process syft message
    reply = node.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    if isinstance(reply, ExceptionMessage):
        return {"error": reply.exception_msg}
    else:
        return [role for role in reply.content]


@router.get("/{role_id}", status_code=200, response_class=JSONResponse)
def get_specific_role_route(
    role_id: int,
    current_user: Any = Depends(get_current_user),
) -> Dict[str, Any]:
    """Retrieves role by its ID.

    Args:
        current_user : Current session.
        role_id: Target role id.
    Returns:
        resp: JSON structure containing target role.
    """

    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = GetRoleMessage(
        address=node.address, role_id=role_id, reply_to=node.address
    ).sign(signing_key=user_key)

    # Process syft message
    reply = node.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    if isinstance(reply, ExceptionMessage):
        return {"error": reply.exception_msg}
    else:
        return reply.content


@router.patch("/{role_id}", status_code=200, response_class=JSONResponse)
def update_use_route(
    role_id: int,
    current_user: Any = Depends(get_current_user),
    name: str = Body(..., example="Researcher"),
    can_make_data_requests: bool = Body(False, example="false"),
    can_triage_data_requests: bool = Body(False, example="false"),
    can_manage_privacy_budget: bool = Body(False, example="false"),
    can_create_users: bool = Body(False, example="false"),
    can_manage_users: bool = Body(False, example="false"),
    can_edit_roles: bool = Body(False, example="false"),
    can_manage_infrastructure: bool = Body(False, example="false"),
    can_upload_data: bool = Body(False, example="false"),
    can_upload_legal_document: bool = Body(False, example="false"),
    can_edit_domain_settings: bool = Body(False, example="false"),
) -> Dict[str, str]:
    """Changes role attributes

    Args:
        current_user : Current session.
        role_id: Target role id.
        name: New role name.
        can_triage_requests: Update triage requests policy.
        can_edit_settings: Update edit settings policy.
        can_create_users: Update create users policy.
        can_create_groups: Update create groups policy.
        can_edit_roles: Update edit roles policy.
        can_manage_infrastructure: Update Node's infrastructure management policy.
        can_upload_data: Update upload data policy.
    Returns:
        resp: JSON structure containing a log message.
    """
    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = UpdateRoleMessage(
        address=node.address,
        role_id=role_id,
        name=name,
        can_make_data_requests=can_make_data_requests,
        can_triage_data_requests=can_triage_data_requests,
        can_manage_privacy_budget=can_manage_privacy_budget,
        can_create_users=can_create_users,
        can_manage_users=can_manage_users,
        can_edit_roles=can_edit_roles,
        can_manage_infrastructure=can_manage_infrastructure,
        can_upload_data=can_upload_data,
        can_upload_legal_document=can_upload_legal_document,
        can_edit_domain_settings=can_edit_domain_settings,
        reply_to=node.address,
    ).sign(signing_key=user_key)

    # Process syft message
    reply = node.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    if isinstance(reply, ExceptionMessage):
        return {"error": reply.exception_msg}
    else:
        return {"message": reply.resp_msg}


@router.delete("/{role_id}", status_code=200, response_class=JSONResponse)
def delete_user_role(
    role_id: int,
    current_user: Any = Depends(get_current_user),
) -> Dict[str, str]:
    """Deletes a user

    Args:
        role_id: Target role_id.
        current_user : Current session.
    Returns:
        resp: JSON structure containing a log message
    """
    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = DeleteRoleMessage(
        address=node.address, role_id=role_id, reply_to=node.address
    ).sign(signing_key=user_key)

    # Process syft message
    reply = node.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    if isinstance(reply, ExceptionMessage):
        return {"error": reply.exception_msg}
    else:
        return {"message": reply.resp_msg}
