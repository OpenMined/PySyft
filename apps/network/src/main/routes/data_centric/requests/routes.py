from json import dumps, loads

from flask import request, Response
from syft.core.node.common.service.repr_service import ReprMessage
from syft.grid.messages.dataset_messages import CreateDatasetMessage
from syft.grid.messages.request_messages import (
    CreateRequestMessage,
    GetRequestMessage,
    GetRequestsMessage,
    UpdateRequestMessage,
    DeleteRequestMessage,
)

from ...auth import error_handler, token_required, optional_token
from main.core.task_handler import route_logic, task_handler
from ..blueprint import dcfl_blueprint as dcfl_route
from ....core.node import node


@dcfl_route.route("/requests", methods=["POST"])
@token_required
def create_request(current_user):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}

    status_code, response_msg = error_handler(
        route_logic, CreateRequestMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@dcfl_route.route("/requests/<request_id>", methods=["GET"])
@token_required
def get_specific_request(current_user, request_id):
    content = {}
    content["request_id"] = request_id

    status_code, response_msg = error_handler(
        route_logic, GetRequestMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@dcfl_route.route("/requests", methods=["GET"])
@token_required
def get_all_requests(current_user):
    content = {}

    status_code, response_msg = error_handler(
        route_logic, GetRequestsMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@dcfl_route.route("/requests/<request_id>", methods=["PUT"])
@token_required
def update_request(current_user, request_id):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}

    content["request_id"] = request_id

    status_code, response_msg = error_handler(
        route_logic, UpdateRequestMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@dcfl_route.route("/requests/<request_id>", methods=["DELETE"])
@token_required
def delete_request(current_user, request_id):
    content = {}
    content["request_id"] = request_id

    status_code, response_msg = error_handler(
        route_logic, DeleteRequestMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        dumps(response),
        status=status_code,
        mimetype="application/json",
    )
