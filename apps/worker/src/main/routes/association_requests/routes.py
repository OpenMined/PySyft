from .blueprint import association_requests_blueprint as association_request_route
from flask import request, Response
import json

from syft.grid.messages.association_messages import (
    SendAssociationRequestMessage,
    ReceiveAssociationRequestMessage,
    GetAssociationRequestMessage,
    DeleteAssociationRequestMessage,
    GetAssociationRequestsMessage,
    RespondAssociationRequestMessage,
)

from ..auth import error_handler, token_required
from ...core.task_handler import route_logic


@association_request_route.route("/request", methods=["POST"])
@token_required
def send_association_request(current_user):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}

    # ADD token parameter
    token = request.headers.get("token", None)
    content["token"] = token

    # ADD sender address parameter
    sender_address = "http://{}".format(request.host)
    content["sender_address"] = sender_address

    status_code, response_msg = error_handler(
        route_logic, SendAssociationRequestMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@association_request_route.route("/receive", methods=["POST"])
def recv_association_request():
    # Get request body
    content = request.get_json()
    if not content:
        content = {}

    status_code, response_msg = error_handler(
        route_logic, ReceiveAssociationRequestMessage, None, content
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
    # Get request body
    content = request.get_json()
    if not content:
        content = {}

    # ADD token parameter
    token = request.headers.get("token", None)
    content["token"] = token

    # ADD sender address parameter
    sender_address = "http://{}".format(request.host)
    content["sender_address"] = sender_address

    status_code, response_msg = error_handler(
        route_logic, RespondAssociationRequestMessage, current_user, content
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
    # Get request body
    content = request.get_json()
    if not content:
        content = {}

    status_code, response_msg = error_handler(
        route_logic, GetAssociationRequestsMessage, current_user, content
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
    # Get request body
    content = request.get_json()
    if not content:
        content = {}

    # ADD path parameters
    content["association_request_id"] = association_request_id

    status_code, response_msg = error_handler(
        route_logic, GetAssociationRequestMessage, current_user, content
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
    # Get request body
    content = request.get_json()
    if not content:
        content = {}

    # ADD path parameters
    content["association_request_id"] = association_request_id

    status_code, response_msg = error_handler(
        route_logic, DeleteAssociationRequestMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )
