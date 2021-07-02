# stdlib
from typing import Any

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
from syft.grid.messages.setup_messages import CreateInitialSetUpMessage
from syft.grid.messages.setup_messages import GetSetUpMessage

# grid absolute
from app.api import deps
from app.core.node import domain

router = APIRouter()


@router.post("", status_code=200, response_class=JSONResponse)
def create_setup(
    email: str = Body(..., example="info@openmined.org"),
    password: str = Body(..., example="changethis"),
    domain_name: str = Body(..., example="OpenGrid"),
) -> Any:
    """
    You must pass valid email,password and domain_name to setup the initial configs.
    """
    # Build Syft Message
    msg = CreateInitialSetUpMessage(
        address=domain.address,
        email=email,
        password=password,
        domain_name=domain_name,
        reply_to=domain.address,
    ).sign(signing_key=domain.signing_key)

    # Process syft message
    reply = domain.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    resp = {}
    if isinstance(reply, ExceptionMessage):
        resp = {"error": reply.exception_msg}
    else:
        resp = {"message": reply.resp_msg}

    return resp


@router.get("", status_code=200, response_class=JSONResponse)
def get_setup(
    request: Request, current_user: Any = Depends(deps.get_current_user)
) -> Any:
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    msg = GetSetUpMessage(address=domain.address, reply_to=domain.address).sign(
        signing_key=user_key
    )

    reply = domain.recv_immediate_msg_with_reply(msg=msg)

    return {"message": reply.message.content}
