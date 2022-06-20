# stdlib
from typing import Any
from typing import Dict

# third party
from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from fastapi.responses import JSONResponse
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey

# syft absolute
from syft.core.node.common.action.exception_action import ExceptionMessage

# syft
from syft.core.node.common.node_service.association_request.association_request_messages import (
    DeleteAssociationRequestMessage,
)
from syft.core.node.common.node_service.association_request.association_request_messages import (
    GetAssociationRequestMessage,
)
from syft.core.node.common.node_service.association_request.association_request_messages import (
    GetAssociationRequestsMessage,
)
from syft.core.node.common.node_service.association_request.association_request_messages import (
    ReceiveAssociationRequestMessage,
)
from syft.core.node.common.node_service.association_request.association_request_messages import (
    RespondAssociationRequestMessage,
)
from syft.core.node.common.node_service.association_request.association_request_messages import (
    SendAssociationRequestMessage,
)

# grid absolute
from grid.api.dependencies.current_user import get_current_user
from grid.core.node import node

router = APIRouter()


@router.post("/request", status_code=200, response_class=JSONResponse)
def send_association_request(
    target: str, source: str, current_user: Any = Depends(get_current_user)
) -> Any:
    """Sends a new association request to the target address
    Args:
        current_user : Current session.
        target: Target address.
        source: Source address.
    Returns:
        resp: JSON structure containing a log message
    """
    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = SendAssociationRequestMessage(
        address=node.address,
        metadata={},
        target=target,
        source=source,
        reply_to=node.address,
    ).sign(signing_key=user_key)

    # Process syft message
    reply = node.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    resp = {}
    if isinstance(reply, ExceptionMessage):
        resp = {"error": reply.exception_msg}
    else:
        resp = {"message": reply.resp_msg}

    return resp


@router.post("/receive", status_code=201, response_class=JSONResponse)
def receive_association_request(
    name: str = Body(..., example="Nodes Association Request"),
    source: str = Body(..., example="http://<node_address>/api/v1"),
    target: str = Body(..., example="http://<target_address>/api/v1"),
) -> Dict[str, str]:
    """Receives a new association request to the sender address
    Args:
        current_user : Current session.
        name: Association request name.
        target: Target address.
        source: Source address.
    Returns:
        resp: JSON structure containing a log message
    """
    # Build Syft Message
    msg = ReceiveAssociationRequestMessage(
        address=node.address,
        name=name,
        source=source,
        target=target,
        reply_to=node.address,
    ).sign(signing_key=SigningKey.generate())

    # Process syft message
    reply = node.send_immediate_msg_without_reply(msg=msg).message

    # Handle Response types
    if isinstance(reply, ExceptionMessage):
        return {"error": reply.exception_msg}
    else:
        return {"message": reply.resp_msg}


@router.post("/reply", status_code=201, response_class=JSONResponse)
def respond_association_request(
    source: str, target: str, current_user: Any = Depends(get_current_user)
) -> Dict[str, str]:
    """Replies an association request

    Args:
        current_user : Current session.
        name: Association request name.
        handshake: Code attached to this association request.
        target: Target address.
        sender: Sender address.
    Returns:
        resp: JSON structure containing a log message
    """
    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = RespondAssociationRequestMessage(
        address=node.address, target=target, source=source, reply_to=node.address
    ).sign(signing_key=user_key)

    # Process syft message
    reply = node.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    if isinstance(reply, ExceptionMessage):
        return {"error": reply.exception_msg}
    else:
        return {"message": reply.resp_msg}


@router.get("", status_code=200, response_class=JSONResponse)
def get_all_association_requests(
    current_user: Any = Depends(get_current_user),
) -> Dict[str, Any]:
    """Retrieves all association requests
    Args:
        current_user : Current session.
    Returns:
        resp: JSON structure containing registered association requests.
    """
    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = GetAssociationRequestsMessage(
        address=node.address, reply_to=node.address
    ).sign(signing_key=user_key)

    # Process syft message
    reply = node.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    if isinstance(reply, ExceptionMessage):
        return {"error": reply.exception_msg}
    else:
        return reply.content


@router.get("/{association_request_id}", status_code=200, response_class=JSONResponse)
def get_specific_association_route(
    association_request_id: int, current_user: Any = Depends(get_current_user)
) -> Dict[str, Any]:
    """Retrieves specific association
    Args:
        current_user : Current session.
        association_request_id: Association request ID.
    Returns:
        resp: JSON structure containing specific association request.
    """
    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = GetAssociationRequestMessage(
        address=node.address,
        association_request_id=association_request_id,
        reply_to=node.address,
    ).sign(signing_key=user_key)

    # Process syft message
    reply = node.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    if isinstance(reply, ExceptionMessage):
        return {"error": reply.exception_msg}
    else:
        return reply.content


@router.delete(
    "/{association_request_id}", status_code=200, response_class=JSONResponse
)
def delete_association_route(
    association_request_id: int, current_user: Any = Depends(get_current_user)
) -> Dict[str, str]:
    """Deletes specific association
    Args:
        current_user : Current session.
        association_request_id: Association request ID.
    Returns:
        resp: JSON structure containing a log message
    """
    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = DeleteAssociationRequestMessage(
        address=node.address,
        association_request_id=association_request_id,
        reply_to=node.address,
    ).sign(signing_key=user_key)

    # Process syft message
    reply = node.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    if isinstance(reply, ExceptionMessage):
        return {"error": reply.exception_msg}
    else:
        return {"message": reply.resp_msg}
