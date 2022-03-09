# stdlib
import json
from typing import Any
from typing import Dict
from typing import List
from typing import Union

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
from syft import deserialize
from syft.core.node.common.action.exception_action import ExceptionMessage
from syft.core.node.common.node_service.dataset_manager.dataset_manager_messages import (
    CreateDatasetMessage,
)
from syft.core.node.common.node_service.dataset_manager.dataset_manager_messages import (
    DeleteDatasetMessage,
)
from syft.core.node.common.node_service.dataset_manager.dataset_manager_messages import (
    GetDatasetMessage,
)
from syft.core.node.common.node_service.dataset_manager.dataset_manager_messages import (
    GetDatasetsMessage,
)
from syft.core.node.common.node_service.dataset_manager.dataset_manager_messages import (
    UpdateDatasetMessage,
)
from syft.lib.python import Dict as SyftDict

# grid absolute
from grid.api.dependencies.current_user import get_current_user
from grid.core.node import node

router = APIRouter()


@router.post("", status_code=201, response_class=JSONResponse)
def upload_dataset_route(
    current_user: Any = Depends(get_current_user),
    file: UploadFile = File(...),
    metadata: str = Form(...),
) -> Dict[str, Any]:
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
        address=node.address,
        dataset=file.file.read(),
        metadata=SyftDict(metadata),
        reply_to=node.address,
    ).sign(signing_key=user_key)

    # Process syft message
    reply = node.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    if isinstance(reply, ExceptionMessage):
        resp = {"error": reply.exception_msg}
    else:
        resp = {"message": reply.resp_msg}

    return resp


@router.get("", status_code=200, response_class=JSONResponse)
def get_all_dataset_metadata_route(
    current_user: Any = Depends(get_current_user),
) -> Union[Dict[str, str], List[Any]]:
    """Retrieves all registered datasets

    Args:
        current_user : Current session.
    Returns:
        resp: JSON structure containing registered datasets.
    """
    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = GetDatasetsMessage(address=node.address, reply_to=node.address).sign(
        signing_key=user_key
    )

    # Process syft message
    reply = node.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    if isinstance(reply, ExceptionMessage):
        return {"error": reply.exception_msg}
    else:
        result = [content for content in reply.metadatas]

        new_all = list()
        for dataset in result:
            new_dataset = {}
            for k, v_blob in dataset.items():
                if k not in ["str_metadata", "blob_metadata", "manifest"]:
                    new_dataset[k] = deserialize(v_blob, from_bytes=True)
            new_all.append(new_dataset)

        return new_all


@router.get("/{dataset_id}", status_code=200, response_class=JSONResponse)
def get_specific_dataset_metadata_route(
    dataset_id: str, current_user: Any = Depends(get_current_user)
) -> Union[Dict[str, str], Any]:
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
        address=node.address, dataset_id=dataset_id, reply_to=node.address
    ).sign(signing_key=user_key)

    # Process syft message
    reply = node.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    if isinstance(reply, ExceptionMessage):
        return {"error": reply.exception_msg}
    else:
        return reply.metadatas.upcast()


@router.put("/{dataset_id}", status_code=200, response_class=JSONResponse)
def update_dataset_metadata_route(
    dataset_id: str,
    current_user: Any = Depends(get_current_user),
    manifest: str = Body(default=None, example="Dataset Manifest"),
    description: str = Body(default=None, example="My brief dataset description ..."),
    tags: list = Body(default=None, example=["#dataset-sample", "#labels"]),
) -> Dict[str, str]:
    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    metadata = {"manifest": manifest, "description": description, "tags": tags}

    # Build Syft Message
    msg = UpdateDatasetMessage(
        address=node.address,
        dataset_id=dataset_id,
        metadata=metadata,
        reply_to=node.address,
    ).sign(signing_key=user_key)

    # Process syft message
    reply = node.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    if isinstance(reply, ExceptionMessage):
        return {"error": reply.exception_msg}
    else:
        return {"message": reply.resp_msg}


@router.delete("/{dataset_id}", status_code=200, response_class=JSONResponse)
def delete_dataset_route(
    dataset_id: str, current_user: Any = Depends(get_current_user)
) -> Dict[str, str]:
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
        address=node.address, dataset_id=dataset_id, reply_to=node.address
    ).sign(signing_key=user_key)

    # Process syft message
    reply = node.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    if isinstance(reply, ExceptionMessage):
        return {"error": reply.exception_msg}
    else:
        return {"message": reply.resp_msg}
