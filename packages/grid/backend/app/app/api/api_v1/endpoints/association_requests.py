# stdlib
from typing import Any

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
from syft.grid.messages.association_messages import DeleteAssociationRequestMessage
from syft.grid.messages.association_messages import GetAssociationRequestMessage
from syft.grid.messages.association_messages import GetAssociationRequestsMessage
from syft.grid.messages.association_messages import ReceiveAssociationRequestMessage
from syft.grid.messages.association_messages import RespondAssociationRequestMessage
from syft.grid.messages.association_messages import SendAssociationRequestMessage

# grid absolute
from app.api import deps
from app.core.node import domain

router = APIRouter()


@router.get("/request", status_code=200, response_class=JSONResponse)
def send_association_request(
    current_user: Any = Depends(deps.get_current_user),
    name: str = Body(..., example="Nodes Association Request"),
    target: str = Body(..., example="http://<target_address>/api/v1"),
    sender: str = Body(..., example="http://<node_address>/api/v1"),
) -> Any:
    ''' Sends a new association request to the target address
        Args:
            current_user : Current session.
            name: Association request name.
            target: Target address.
            sender: Sender address.
        Returns:
            resp: JSON structure containing a log message
    '''
    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = SendAssociationRequestMessage(
        address=domain.address,
        name=name,
        target=target,
        sender=sender,
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


@router.post("/receive", status_code=201, response_class=JSONResponse)
def receive_association_request(
    current_user: Any = Depends(deps.get_current_user),
    name: str = Body(..., example="Nodes Association Request"),
    handshake: str = Body(..., example="<hash_code>"),
    sender: str = Body(..., example="http://<node_address>/api/v1"),
    target: str = Body(..., example="http://<target_address>/api/v1"),
):
    ''' Receives a new association request to the sender address
        Args:
            current_user : Current session.
            name: Association request name.
            handshake: Code attached to this association request.
            target: Target address.
            sender: Sender address.
        Returns:
            resp: JSON structure containing a log message
    '''
    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = ReceiveAssociationRequestMessage(
        address=domain.address,
        handshake=handshake,
        sender=sender,
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


@router.post("/reply", status_code=201, response_class=JSONResponse)
def respond_association_request(
    current_user: Any = Depends(deps.get_current_user),
    handshake: str = Body(..., example="<hash_code>"),
    value: str = Body(..., example="<hash_code>"),
    target: str = Body(..., example="http://<target_address>/api/v1"),
    sender: str = Body(..., example="http://<node_address>/api/v1"),
):
    ''' Replies an association request

        Args:
            current_user : Current session.
            name: Association request name.
            handshake: Code attached to this association request.
            target: Target address.
            sender: Sender address.
        Returns:
            resp: JSON structure containing a log message
    '''
    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = RespondAssociationRequestMessage(
        address=domain.address,
        value=value,
        handshake=handshake,
        target=target,
        sender=sender,
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
def get_all_association_requests(
    current_user: Any = Depends(deps.get_current_user),
):
    ''' Retrieves all association requests
        Args:
            current_user : Current session.
        Returns:
            resp: JSON structure containing registered association requests.
    '''
    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = GetAssociationRequestsMessage(
        address=domain.address, reply_to=domain.address
    ).sign(signing_key=user_key)

    # Process syft message
    reply = domain.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    resp = {}
    if isinstance(reply, ExceptionMessage):
        resp = {"error": reply.exception_msg}
    else:
        resp = reply.content

    return resp


@router.get("/{association_request_id}", status_code=200, response_class=JSONResponse)
def get_specific_association_route(
    association_request_id: int,
    current_user: Any = Depends(deps.get_current_user),
):
    ''' Retrieves specific association
        Args:
            current_user : Current session.
            association_request_id: Association request ID.
        Returns:
            resp: JSON structure containing specific association request.
    '''
    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = GetAssociationRequestMessage(
        address=domain.address,
        association_request_id=association_request_id,
        reply_to=domain.address,
    ).sign(signing_key=user_key)

    # Process syft message
    reply = domain.recv_immediate_msg_with_reply(msg=msg).message

    # Handle Response types
    resp = {}
    if isinstance(reply, ExceptionMessage):
        resp = {"error": reply.exception_msg}
    else:
        resp = reply.content

    return resp


@router.delete(
    "/{association_request_id}", status_code=200, response_class=JSONResponse
)
def delete_association_route(
    association_request_id: int,
    current_user: Any = Depends(deps.get_current_user),
):
    ''' Deletes specific association
        Args:
            current_user : Current session.
            association_request_id: Association request ID.
        Returns:
            resp: JSON structure containing a log message
    '''
    # Map User Key
    user_key = SigningKey(current_user.private_key.encode(), encoder=HexEncoder)

    # Build Syft Message
    msg = DeleteAssociationRequestMessage(
        address=domain.address,
        association_request_id=association_request_id,
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
