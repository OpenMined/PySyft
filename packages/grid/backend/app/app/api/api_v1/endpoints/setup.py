from typing import Any 

# third party
from fastapi import APIRouter
from fastapi import Request
from fastapi import Response
import json

# syft
from syft.grid.messages.setup_messages import CreateInitialSetUpMessage


from app.core.node import domain
router = APIRouter()


@router.post("", response_model=str)
async def syft_metadata( 
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
