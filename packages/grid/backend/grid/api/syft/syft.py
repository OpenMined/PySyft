# stdlib
import json
from typing import Any

# third party
from fastapi import APIRouter
from fastapi import Depends
from fastapi import Request
from fastapi import Response
from fastapi.responses import JSONResponse

# syft absolute
from syft import __version__
from syft import deserialize
from syft import serialize
from syft.core.common.message import SignedImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.common.message import SignedMessage
from syft.core.node.enums import RequestAPIFields
from syft.util import get_tracer

# grid absolute
from grid.api.dependencies.current_user import get_current_user
from grid.api.users.models import UserPrivate
from grid.core.celery_app import celery_app
from grid.core.config import settings
from grid.core.node import node

router = APIRouter()

tracer = get_tracer("API")


async def get_body(request: Request) -> bytes:
    return await request.body()


@router.get("/version")
def syft_version() -> Response:
    return JSONResponse(content={"version": __version__})


@router.get("/metadata", response_model=str)
def syft_metadata() -> Response:
    return Response(
        node.get_metadata_for_client()._object2proto().SerializeToString(),
        media_type="application/octet-stream",
    )


@router.delete("", response_model=str)
def delete(current_user: UserPrivate = Depends(get_current_user)) -> Response:
    # If current user is the node owner ...
    success = node.clear(current_user.role)
    if success:
        response = {RequestAPIFields.MESSAGE: "Domain node has been reset!"}
    else:
        response = {
            RequestAPIFields.ERROR: "You're not allowed to reset the node data."
        }

    return Response(json.dumps(response))


@router.post("", response_model=str)
def syft_route(data: bytes = Depends(get_body)) -> Any:
    print("got a new incoming request")
    with tracer.start_as_current_span("POST syft_route"):
        obj_msg = deserialize(blob=data, from_bytes=True)
        is_isr = isinstance(obj_msg, SignedImmediateSyftMessageWithReply) or isinstance(
            obj_msg, SignedMessage
        )
        if is_isr:
            reply = node.recv_immediate_msg_with_reply(msg=obj_msg)
            r = Response(
                serialize(obj=reply, to_bytes=True),
                media_type="application/octet-stream",
            )
            return r
        elif isinstance(obj_msg, SignedImmediateSyftMessageWithoutReply):
            celery_app.send_task("grid.worker.msg_without_reply", args=[obj_msg])
        else:
            node.recv_eventual_msg_without_reply(msg=obj_msg)
        return ""


@router.post("/stream", response_model=str)
def syft_stream(data: bytes = Depends(get_body)) -> Any:
    with tracer.start_as_current_span("POST syft_route /stream"):
        if settings.STREAM_QUEUE:
            print("Queuing streaming message for processing on worker node")
            try:
                # we pass in the bytes and they get handled by the custom serde code
                # inside celery_app.py
                celery_app.send_task("grid.worker.msg_without_reply", args=[data])
            except Exception as e:
                print(f"Failed to queue work on streaming endpoint. {type(data)}. {e}")
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
