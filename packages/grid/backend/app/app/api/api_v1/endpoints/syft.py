from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.orm import Session

from app import crud, models, schemas
from app.api import deps

import json

router = APIRouter()

import syft as sy
from syft.core.common.message import SignedImmediateSyftMessageWithReply
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply

domain = sy.Domain("my domain")

@router.get("/", response_model=str)
def read_items(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Retrieve items.
    """
    if crud.user.is_superuser(current_user):
        items = crud.item.get_multi(db, skip=skip, limit=limit)
    else:
        items = crud.item.get_multi_by_owner(
            db=db, owner_id=current_user.id, skip=skip, limit=limit
        )
    print("SOME ITEMS")
    print(items)
    return "hello world"

# @router.get("/metadata", response_model=str)
# def metadata_route():
#     # grid relative
#     response_body = {
#         "metadata": sy.serialize(domain.get_metadata_for_client())
#         .SerializeToString()
#         .decode("ISO-8859-1")
#     }
#     return json.dumps(response_body)


@router.post("/msg", response_model=str)
async def msg(
    request: Request,
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:

    data = await request.body()
    obj_msg = sy.deserialize(blob=data, from_bytes=True)
    if isinstance(obj_msg, SignedImmediateSyftMessageWithReply):
        reply = domain.recv_immediate_msg_with_reply(msg=obj_msg)
        r = Response(response=sy.serialize(obj=reply, to_bytes=True), status=200)
        r.headers["Content-Type"] = "application/octet-stream"
        return r
    elif isinstance(obj_msg, SignedImmediateSyftMessageWithoutReply):
        domain.recv_immediate_msg_without_reply(msg=obj_msg)
    else:
        domain.recv_eventual_msg_without_reply(msg=obj_msg)

    #
    # if crud.user.is_superuser(current_user):
    #     items = crud.item.get_multi(db, skip=skip, limit=limit)
    # else:
    #     items = crud.item.get_multi_by_owner(
    #         db=db, owner_id=current_user.id, skip=skip, limit=limit

    return "sending message to domain"
