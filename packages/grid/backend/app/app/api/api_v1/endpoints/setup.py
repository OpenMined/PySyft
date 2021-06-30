from typing import Any 

# third party
from fastapi import APIRouter
from fastapi import Request
from fastapi import Response
from fastapi import Depends
import json
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey

# syft
from syft.grid.messages.setup_messages import CreateInitialSetUpMessage
from syft.grid.messages.setup_messages import GetSetUpMessage


from app.core.node import domain
from app.api import deps

router = APIRouter()


@router.post("", response_model=str)
async def create_setup( 
    request: Request,
) -> Any:
    data = json.loads( await request.body())

    msg = CreateInitialSetUpMessage(
        address=domain.address,
        content=data,
        reply_to=domain.address
    ).sign(signing_key=domain.signing_key)

    reply = domain.recv_immediate_msg_with_reply(msg=msg)

    return Response(
            json.dumps({"message": reply.message.content}),
            media_type="application/json",
    )


@router.get("", response_model=str)
def get_setup( 
    request: Request,
    current_user: Any = Depends(deps.get_current_user)
) -> Any:
    user_key =  SigningKey(
        current_user.private_key.encode(),
        encoder=HexEncoder
        )
    
    msg = GetSetUpMessage(
        address=domain.address,
        reply_to=domain.address
    ).sign(signing_key=user_key)

    reply = domain.recv_immediate_msg_with_reply(msg=msg)

    return Response(
            json.dumps({"message": reply.message.content}),
            media_type="application/json",
    )