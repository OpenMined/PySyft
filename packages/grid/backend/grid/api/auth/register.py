# stdlib
from typing import Any

# third party
from fastapi import APIRouter
from fastapi import Body
from fastapi.responses import JSONResponse

# syft absolute
from syft import flags
from syft.core.node.common.node_manager.role_manager import RoleManager
from syft.core.node.common.node_manager.user_manager import UserManager

# grid absolute
from grid.api.users.models import UserPrivate
from grid.core.config import settings
from grid.core.node import node
from grid.utils import send_message_with_reply

if flags.USE_NEW_SERVICE:
    # syft absolute
    from syft.core.node.common.node_service.user_manager.new_user_messages import (
        CreateUserMessage,
    )
else:
    # syft absolute
    from syft.core.node.common.node_service.user_manager.user_manager_service import (
        CreateUserMessage,
    )

router = APIRouter()


@router.post("/register", name="register", status_code=200, response_class=JSONResponse)
def register(data: dict = Body(..., example="sheldon@caltech.edu")) -> Any:

    if not settings.OPEN_REGISTRATION:
        return {
            "error": "This node doesn't allow for anyone to register a user. Please contact the domain"
            + " administrator who can setup a user account for you."
        }

    if "name" not in data:
        return {"error": "Missing 'name' attribute. Please try again"}

    if "email" not in data:
        return {"error": "Missing 'email' attribute. Please try again"}

    if "password" not in data:
        return {"error": "Missing 'password' attribute. Please try again"}

    new_user = {
        "name": data["name"],
        "email": data["email"],
        "password": data["password"],
        "institution": data.get("institution"),
        "website": data.get("website"),
        "budget": 0.0,
    }

    owner_role = RoleManager(node.db_engine).owner_role
    root_user = UserManager(node.db_engine).first(role=owner_role.id)
    root_user = UserPrivate.from_orm(root_user)

    reply = send_message_with_reply(
        signing_key=root_user.get_signing_key(),
        message_type=CreateUserMessage,
        **dict(new_user)
    )

    if flags.USE_NEW_SERVICE:
        return reply.message
    return reply.resp_msg
