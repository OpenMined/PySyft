# stdlib
import json
from typing import Any

# third party
from fastapi import APIRouter
from fastapi import Depends
from fastapi import Request
from fastapi import Response
from starlette import status
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey

# grid absolute
from grid.core.node import node
from grid.core.config import settings
from grid.api.dependencies.current_user import get_current_user
from grid.utils import send_message_with_reply

# syft absolute
from syft.core.node.common.node_service.oblv.oblv_service import (
    CreateKeyPairMessage, GetPublicKeyMessage, PublishDatasetMessage
)

from syft.core.node.common.action.exception_action import ExceptionMessage
router = APIRouter()

async def get_body(request: Request) -> bytes:
    return await request.body()

@router.post("/key", name="key:generate", status_code=status.HTTP_200_OK)
def generate_oblv_key_pair(
    current_user: Any = Depends(get_current_user),
):
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = CreateKeyPairMessage(
        address=node.address, reply_to=node.address
    ).sign(signing_key=user_key)

    # Process syft message
    reply = node.recv_immediate_msg_with_reply(msg=msg).message
    
    # Handle Response types
    if isinstance(reply, ExceptionMessage):
        return {"error": reply.exception_msg}
    else:
        return reply.resp_msg


@router.get("/key", name="key:get", status_code=status.HTTP_200_OK)
def get_public_key(
    current_user: Any = Depends(get_current_user)
):
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = GetPublicKeyMessage(
        address=node.address, reply_to=node.address
    ).sign(signing_key=user_key)
    reply = node.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    if isinstance(reply, ExceptionMessage):
        return {"error": reply.exception_msg}
    else:
        return reply.response
    
@router.post("/publish", name="publish:post", status_code=status.HTTP_200_OK)
def publish_dataset(
    deployment_id: str,
    dataset_id: str,
    current_user: Any = Depends(get_current_user),
):
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)
    return "Success"