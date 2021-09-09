# stdlib
from typing import Any
from typing import Optional

# third party
from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from fastapi import Request
from fastapi.responses import JSONResponse
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey

# syft absolute
from syft.core.node.common.action.exception_action import ExceptionMessage

# syft
from syft.core.node.common.node_service.node_setup.node_setup_messages import (
    UpdateSetupMessage,
)
from syft.core.node.common.node_service.node_setup.node_setup_service import (
    GetSetUpMessage,
)

# grid absolute
from grid.api.dependencies.current_user import get_current_user
from grid.core.node import node

router = APIRouter()


@router.post("", status_code=200, response_class=JSONResponse)
def update_settings(
    contact: str = Body(..., example="contact@openmined.org"),
    domain_name: str = Body(..., example="OpenGrid"),
    description: str = Body(..., example="A Domain Node Description ..."),
    daa: Optional[bool] = False,
) -> Any:
    """
    You must pass valid email,password and domain_name to setup the initial configs.
    """

    # Build Syft Message
    msg = UpdateSetupMessage(
        address=node.address,
        contact=contact,
        domain_name=domain_name,
        description=description,
        daa=daa,
        reply_to=node.address,
    ).sign(signing_key=node.signing_key)

    # Process syft message
    reply = node.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    resp = {}
    if isinstance(reply, ExceptionMessage):
        resp = {"error": reply.exception_msg}
    else:
        resp = {"message": reply.resp_msg}
    return resp


@router.get("", status_code=200, response_class=JSONResponse)
def get_setup(request: Request, current_user: Any = Depends(get_current_user)) -> Any:
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    msg = GetSetUpMessage(address=node.address, reply_to=node.address).sign(
        signing_key=user_key
    )

    reply = node.recv_immediate_msg_with_reply(msg=msg)

    return {"message": reply.message.content}
