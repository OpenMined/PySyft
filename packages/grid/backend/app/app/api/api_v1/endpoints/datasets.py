# stdlib
import io
import json
from typing import Any
from typing import List
from typing import Optional

# third party
from fastapi import APIRouter
from fastapi import Body
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
from syft.grid.messages.dataset_messages import CreateDatasetMessage
from syft.grid.messages.dataset_messages import DeleteDatasetMessage
from syft.grid.messages.dataset_messages import GetDatasetMessage
from syft.grid.messages.dataset_messages import GetDatasetResponse
from syft.grid.messages.dataset_messages import GetDatasetsMessage
from syft.grid.messages.dataset_messages import GetDatasetsResponse
from syft.grid.messages.dataset_messages import UpdateDatasetMessage
from syft.lib.python import Dict as SyftDict

# grid absolute
from app.api import deps
from app.core.node import domain

router = APIRouter()


@router.post("", status_code=201, response_class=JSONResponse)
def upload_dataset_route(
    current_user: Any = Depends(deps.get_current_user),
    file: UploadFile = File(...),
    metadata: str = Form(...),
):
    """Upload a compressed dataset file(s)

    Args:
        current_user : Current session.
        file: Compressed file (tar.gz) containing different csv files.
        metadata: Metadata related to the compressed file.
    Returns:
        resp: JSON structure containing a log message.
    """
    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)
    metadata = json.loads(metadata)

    msg = CreateDatasetMessage(
        address=domain.address,
        dataset=file.file.read(),
        metadata=SyftDict(metadata),
        reply_to=domain.address,
    ).sign(signing_key=user_key)

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
def get_all_dataset_metadata_route(
    current_user: Any = Depends(deps.get_current_user),
):
    """Retrieves all registered datasets

    Args:
        current_user : Current session.
    Returns:
        resp: JSON structure containing registered datasets.
    """
    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = GetDatasetsMessage(address=domain.address, reply_to=domain.address).sign(
        signing_key=user_key
    )

    # Process syft message
    reply = domain.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    resp = {}
    if isinstance(reply, ExceptionMessage):
        resp = {"error": reply.exception_msg}
    else:
        resp = [dataset.upcast() for dataset in reply.content]

    return resp


@router.get("/{dataset_id}", status_code=200, response_class=JSONResponse)
def get_specific_dataset_metadata_route(
    dataset_id: str,
    current_user: Any = Depends(deps.get_current_user),
):
    """Retrieves dataset by its ID.

    Args:
        current_user : Current session.
        dataset_id: Target dataset id.
    Returns:
        resp: JSON structure containing target dataset.
    """

    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = GetDatasetMessage(
        address=domain.address, dataset_id=dataset_id, reply_to=domain.address
    ).sign(signing_key=user_key)

    # Process syft message
    reply = domain.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    resp = {}
    if isinstance(reply, ExceptionMessage):
        resp = {"error": reply.exception_msg}
    else:
        resp = reply.content.upcast()

    return resp


@router.put("/{dataset_id}", status_code=200, response_class=JSONResponse)
def update_dataset_metadata_route(
    dataset_id: str,
    current_user: Any = Depends(deps.get_current_user),
    manifest: str = Body(default=None, example="Dataset Manifest"),
    description: str = Body(default=None, example="My brief dataset description ..."),
    tags: list = Body(default=None, example=["#dataset-sample", "#labels"]),
):
    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    metadata = SyftDict(
        {
            "manifest": manifest,
            "description": description,
            "tags": tags,
        }
    )

    # Build Syft Message
    msg = UpdateDatasetMessage(
        address=domain.address,
        dataset_id=dataset_id,
        metadata=metadata,
        reply_to=domain.address,
    ).sign(signing_key=user_key)

    # Process syft message
    reply = domain.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    resp = {}
    if isinstance(reply, ExceptionMessage):
        resp = {"error": reply.exception_msg}
    else:
        resp = {"message": reply.resp_msg}

    return resp


@router.delete("/{dataset_id}", status_code=200, response_class=JSONResponse)
def delete_dataset_route(
    dataset_id: str,
    current_user: Any = Depends(deps.get_current_user),
):
    """Deletes a dataset

    Args:
        dataset_id: Target dataset_id.
        current_user : Current session.
    Returns:
        resp: JSON structure containing a log message
    """
    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = DeleteDatasetMessage(
        address=domain.address, dataset_id=dataset_id, reply_to=domain.address
    ).sign(signing_key=user_key)

    # Process syft message
    reply = domain.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    resp = {}
    if isinstance(reply, ExceptionMessage):
        resp = {"error": reply.exception_msg}
    else:
        resp = {"message": reply.resp_msg}

    return resp
