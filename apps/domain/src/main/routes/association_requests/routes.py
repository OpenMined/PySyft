from .blueprint import association_requests_blueprint as association_request_route
from flask import request, Response
import json

from syft.core.node.common.service.repr_service import ReprMessage
from syft.grid.messages.association_messages import (
    SendAssociationRequestMessage,
    ReceiveAssociationRequestMessage,
    GetAssociationRequestMessage,
    DeleteAssociationRequestMessage,
    GetAllAssociationRequestMessage,
    RespondAssociationRequestMessage,
)

from ..auth import error_handler, token_required

# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey

from ...core.node import node
from ...core.task_handler import task_handler, process_as_syft_message
from ...core.exceptions import (
    PyGridError,
    UserNotFoundError,
    RoleNotFoundError,
    GroupNotFoundError,
    AuthorizationError,
    MissingRequestKeyError,
    InvalidCredentialsError,
)


def route_logic(message_class, current_user):
    user_key = SigningKey(current_user.private_key.encode("utf-8"), encoder=HexEncoder)

    # Get request body
    content = request.get_json()
    content = {
        "address": node.address,
        "content": content,
        "reply_to": node.address,
    }

    syft_message = {}
    syft_message["message_class"] = message_class
    syft_message["message_content"] = content
    syft_message["sign_key"] = user_key

    # Execute task
    response_msg = task_handler(
        route_function=process_as_syft_message,
        data=syft_message,
        mandatory={
            "message_class": MissingRequestKeyError,
            "message_content": MissingRequestKeyError,
            "sign_key": MissingRequestKeyError,
        },
    )
    return response_msg


@association_request_route.route("/request", methods=["POST"])
@token_required
def send_association_request(current_user):
    status_code, response_msg = error_handler(
        route_logic, SendAssociationRequestMessage, current_user
    )
    response = response_msg if isinstance(response_msg, dict) else response_msg.content
    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@association_request_route.route("/receive", methods=["POST"])
@token_required
def recv_association_request(current_user):
    status_code, response_msg = error_handler(
        route_logic, ReceiveAssociationRequestMessage, current_user
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@association_request_route.route("/respond", methods=["POST"])
@token_required
def reply_association_request(current_user):
    status_code, response_msg = error_handler(
        route_logic, RespondAssociationRequestMessage, current_user
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@association_request_route.route("/", methods=["GET"])
@token_required
def get_all_association_requests(current_user):
    status_code, response_msg = error_handler(
        route_logic, GetAllAssociationRequestMessage, current_user
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@association_request_route.route("/<association_request_id>", methods=["GET"])
@token_required
def get_specific_association_requests(current_user, association_request_id):
    status_code, response_msg = error_handler(
        route_logic, GetAssociationRequestMessage, current_user
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@association_request_route.route("/<association_request_id>", methods=["DELETE"])
@token_required
def delete_association_requests(current_user, association_request_id):
    status_code, response_msg = error_handler(
        route_logic, DeleteAssociationRequestMessage, current_user
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )
