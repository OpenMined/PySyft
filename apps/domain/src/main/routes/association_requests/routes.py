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


def route_logic(message_class):
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
    syft_message[
        "sign_key"
    ] = node.signing_key  # TODO: Method to map token into sign-key

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
# @token_required
def send_association_request():
    status_code, response_msg = error_handler(
        route_logic, SendAssociationRequestMessage
    )
    return Response(
        json.dumps(response_msg.content),
        status=status_code,
        mimetype="application/json",
    )


@association_request_route.route("/receive", methods=["POST"])
# @token_required
def recv_association_request():
    status_code, response_msg = error_handler(
        route_logic, ReceiveAssociationRequestMessage
    )

    return Response(
        json.dumps(response_msg.content),
        status=status_code,
        mimetype="application/json",
    )


@association_request_route.route("/respond", methods=["POST"])
# @token_required
def reply_association_request():
    status_code, response_msg = error_handler(
        route_logic, RespondAssociationRequestMessage
    )

    return Response(
        json.dumps(response_msg.content),
        status=status_code,
        mimetype="application/json",
    )


@association_request_route.route("/", methods=["GET"])
# @token_required
def get_all_association_requests():
    status_code, response_msg = error_handler(
        route_logic, GetAllAssociationRequestMessage
    )

    return Response(
        json.dumps(response_msg.content),
        status=status_code,
        mimetype="application/json",
    )


@association_request_route.route("/<association_request_id>", methods=["GET"])
# @token_required
def get_specific_association_requests(association_request_id):
    status_code, response_msg = error_handler(route_logic, GetAssociationRequestMessage)

    return Response(
        json.dumps(response_msg.content),
        status=status_code,
        mimetype="application/json",
    )


@association_request_route.route("/<association_request_id>", methods=["DELETE"])
# @token_required
def delete_association_requests(association_request_id):
    status_code, response_msg = error_handler(
        route_logic, DeleteAssociationRequestMessage
    )

    return Response(
        json.dumps(response_msg.content),
        status=status_code,
        mimetype="application/json",
    )
