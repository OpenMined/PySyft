# stdlib
from typing import Any

# third party
from fastapi import APIRouter
from fastapi import Request
from fastapi import Response

# syft absolute
from syft.core.common.message import SignedImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft import serialize
from syft import deserialize

# grid absolute
from app.core.node import domain

router = APIRouter()

@router.get("/metadata", response_model=str)
async def syft_metadata():
    return Response(
        domain.get_metadata_for_client()._object2proto().SerializeToString(),
        media_type="application/octet-stream",
    )

@router.post("", response_model=str)
async def syft( request: Request,
    #    skip: int = 0,
    #    limit: int = 100,
    #    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:

    data = await request.body()
    obj_msg = deserialize(blob=data, from_bytes=True)
    if isinstance(obj_msg, SignedImmediateSyftMessageWithReply):
        reply = domain.recv_immediate_msg_with_reply(msg=obj_msg)
        r = Response(
            serialize(obj=reply, to_bytes=True),
            media_type="application/octet-stream",
        )
        return r
    elif isinstance(obj_msg, SignedImmediateSyftMessageWithoutReply):
        domain.recv_immediate_msg_without_reply(msg=obj_msg)
    else:
        domain.recv_eventual_msg_without_reply(msg=obj_msg)
    return ""