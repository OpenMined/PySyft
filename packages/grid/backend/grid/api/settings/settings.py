# stdlib
import json
from typing import Any
from typing import Optional

# third party
from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import Form
from fastapi import UploadFile
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
from grid.api.users.models import UserPrivate
from grid.core.node import node

router = APIRouter()


@router.post("", status_code=200, response_class=JSONResponse)
def update_settings(
    current_user: UserPrivate = Depends(get_current_user),
    file: Optional[UploadFile] = File(...),
    settings: str = Form(...),
) -> Any:
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    if file:
        pdf_file = file.file.read()  # type: ignore
    else:
        pdf_file = b""

    dict_settings = json.loads(settings)

    # Build Syft Message
    msg = UpdateSetupMessage(
        address=node.address,
        contact=dict_settings.get("contact", ""),
        domain_name=dict_settings.get("domain_name", ""),
        description=dict_settings.get("description", ""),
        daa=dict_settings.get("daa", False),
        daa_document=pdf_file,
        reply_to=node.address,
    ).sign(signing_key=user_key)

    # Process syft message
    reply = node.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    resp = {}
    if isinstance(reply, ExceptionMessage):
        resp = {"error": reply.exception_msg}
    else:
        resp = {"message": reply.content}
    return resp


@router.get("", status_code=200, response_class=JSONResponse)
def get_setup() -> Any:
    msg = GetSetUpMessage(address=node.address, reply_to=node.address).sign(
        signing_key=node.signing_key
    )

    reply = node.recv_immediate_msg_with_reply(msg=msg)
    return JSONResponse(reply.message.content)
