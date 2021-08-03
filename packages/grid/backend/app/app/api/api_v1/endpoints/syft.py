# stdlib
from typing import Any

# third party
from fastapi import APIRouter
from fastapi import Request
from fastapi import Response

# syft absolute
from syft import deserialize  # type: ignore
from syft import serialize  # type: ignore
from syft.core.common.message import SignedImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply

# grid absolute
from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.node import node

router = APIRouter()


@router.get("/metadata", response_model=str)
async def syft_metadata() -> Response:
    return Response(
        node.get_metadata_for_client()._object2proto().SerializeToString(),
        media_type="application/octet-stream",
    )


@router.post("", response_model=str)
async def syft_route(
    request: Request,
    #    skip: int = 0,
    #    limit: int = 100,
    #    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    data = await request.body()
    obj_msg = deserialize(blob=data, from_bytes=True)
    if isinstance(obj_msg, SignedImmediateSyftMessageWithReply):
        reply = node.recv_immediate_msg_with_reply(msg=obj_msg)
        r = Response(
            serialize(obj=reply, to_bytes=True),
            media_type="application/octet-stream",
        )
        return r
    elif isinstance(obj_msg, SignedImmediateSyftMessageWithoutReply):
        node.recv_immediate_msg_without_reply(msg=obj_msg)
    else:
        node.recv_eventual_msg_without_reply(msg=obj_msg)
    return ""


@router.post("/stream", response_model=str)
async def syft_stream(
    request: Request,
) -> Any:
    data = await request.body()

    if settings.STREAM_QUEUE:
        print("Queuing streaming message for processing on worker node")
        # use latin-1 instead of utf-8 because our bytes might not be an even number
        msg_bytes_str = data.decode("latin-1")
        try:
            celery_app.send_task("app.worker.msg_without_reply", args=[msg_bytes_str])
        except Exception:
            print(f"Failed to queue work on streaming endpoint. {msg_bytes_str}")
    else:
        print("Processing streaming message on web node")
        obj_msg = deserialize(blob=data, from_bytes=True)
        if isinstance(obj_msg, SignedImmediateSyftMessageWithReply):
            raise Exception("MessageWithReply not supported on the stream endpoint")
        elif isinstance(obj_msg, SignedImmediateSyftMessageWithoutReply):
            node.recv_immediate_msg_without_reply(msg=obj_msg)
        else:
            raise Exception("MessageWithReply not supported on the stream endpoint")
    return ""
