# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
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
from syft.core.node.common.node_service.user_manager.user_manager_service import (
    CreateUserMessage,
)
from syft.core.node.common.node_service.user_manager.user_manager_service import (
    DeleteUserMessage,
)
from syft.core.node.common.node_service.user_manager.user_manager_service import (
    GetUserMessage,
)
from syft.core.node.common.node_service.user_manager.user_manager_service import (
    GetUsersMessage,
)
from syft.core.node.common.node_service.user_manager.user_manager_service import (
    SearchUsersMessage,
)
from syft.core.node.common.node_service.user_manager.user_manager_service import (
    UpdateUserMessage,
)
from syft.core.node.common.node_table.utils import model_to_json

# grid absolute
from app.api import deps
from app.core.node import node

router = APIRouter()


@router.get("/me", status_code=200, response_class=JSONResponse)
def me_route(current_user: Any = Depends(deps.get_current_user)) -> Any:
    """Returns Current User Table"""
    user_dict = model_to_json(current_user)
    user_dict["role"] = node.roles.first(id=user_dict["role"]).name
    del user_dict["private_key"]
    return user_dict


@router.post("", status_code=201, response_class=JSONResponse)
def create_user(
    current_user: Any = Depends(deps.get_current_user),
    email: str = Body(..., example="info@openmined.org"),
    password: str = Body(..., example="changethis"),
    role: Optional[str] = Body(..., example="User"),
) -> Dict[str, str]:
    """Creates new user user

    Args:
        current_user : Current session.
        email: User email.
        password: User password.
        role: User role name.

    Returns:
        resp: JSON structure containing a log message
    """
    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = CreateUserMessage(
        address=node.address,
        email=email,
        password=password,
        role=role,
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
def get_all_users_route(
    current_user: Any = Depends(deps.get_current_user),
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Retrieves all registered users

    Args:
        current_user : Current session.
    Returns:
        resp: JSON structure containing registered users.
    """
    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = GetUsersMessage(address=node.address, reply_to=node.address).sign(
        signing_key=user_key
    )

    # Process syft message
    reply = node.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    if isinstance(reply, ExceptionMessage):
        return {"error": reply.exception_msg}
    else:
        return [user.upcast() for user in reply.content]


@router.get("/{user_id}", status_code=200, response_class=JSONResponse)
def get_specific_user_route(
    user_id: int,
    current_user: Any = Depends(deps.get_current_user),
) -> Dict[str, Any]:
    """Creates new user user

    Args:
        current_user : Current session.
        user_id: Target user id.
    Returns:
        resp: JSON structure containing target user.
    """

    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = GetUserMessage(
        address=node.address, user_id=user_id, reply_to=node.address
    ).sign(signing_key=user_key)

    # Process syft message
    reply = node.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    if isinstance(reply, ExceptionMessage):
        return {"error": reply.exception_msg}
    else:
        return reply.content.upcast()


@router.patch("/{user_id}", status_code=200, response_class=JSONResponse)
def update_use_route(
    user_id: int,
    current_user: Any = Depends(deps.get_current_user),
    email: Optional[str] = Body(default=None, example="info@openmined.org"),
    password: Optional[str] = Body(default=None, example="changethis"),
    role: Optional[str] = Body(default=None, example="User"),
) -> Dict[str, str]:
    """Changes user attributes

    Args:
        current_user : Current session.
        user_id: Target user id.
        email: New email.
        password: New password.
        role: New role name.

    Returns:
        resp: JSON structure containing a log message
    """
    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = UpdateUserMessage(
        address=node.address,
        user_id=user_id,
        email=email,
        password=password,
        role=role,
        reply_to=node.address,
    ).sign(signing_key=user_key)

    # Process syft message
    reply = node.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    if isinstance(reply, ExceptionMessage):
        return {"error": reply.exception_msg}
    else:
        return {"message": reply.resp_msg}


@router.delete("/{user_id}", status_code=200, response_class=JSONResponse)
def delete_user_role(
    user_id: int,
    current_user: Any = Depends(deps.get_current_user),
) -> Dict[str, str]:
    """Deletes a user

    Args:
        user_id: Target user id.
        current_user : Current session.

    Returns:
        resp: JSON structure containing a log message
    """
    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = DeleteUserMessage(
        address=node.address, user_id=user_id, reply_to=node.address
    ).sign(signing_key=user_key)

    # Process syft message
    reply = node.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    if isinstance(reply, ExceptionMessage):
        return {"error": reply.exception_msg}
    else:
        return {"message": reply.resp_msg}


@router.post("/search", status_code=200, response_class=JSONResponse)
def search_users_route(
    current_user: Any = Depends(deps.get_current_user),
    email: Optional[str] = Body(default=None, example="info@openmined.org"),
    groups: Optional[str] = Body(default=None, example="OM Group"),
    role: Optional[str] = Body(default=None, example="User"),
) -> Dict[str, Any]:
    """Filter users by using it's properties

    Args:
        current_user : Current session.
        email: Filter email.
        role: Filter role name.
        groups: Filter group name.

    Returns:
        resp: JSON structure containing search results.
    """
    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = SearchUsersMessage(
        address=node.address,
        email=email,
        groups=groups,
        role=role,
        reply_to=node.address,
    ).sign(signing_key=user_key)

    # Process syft message
    reply = node.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    if isinstance(reply, ExceptionMessage):
        return {"error": reply.exception_msg}
    else:
        return reply.content
