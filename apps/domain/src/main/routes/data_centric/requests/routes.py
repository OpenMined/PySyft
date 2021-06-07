# stdlib
from json import dumps
from json import loads

# third party
from flask import Response
from flask import request
from main.core.task_handler import route_logic
from main.core.task_handler import task_handler
from syft.core.node.common.service.repr_service import ReprMessage
from syft.grid.messages.dataset_messages import CreateDatasetMessage
from syft.grid.messages.request_messages import CreateRequestMessage
from syft.grid.messages.request_messages import DeleteRequestMessage
from syft.grid.messages.request_messages import GetRequestMessage
from syft.grid.messages.request_messages import GetRequestsMessage
from syft.grid.messages.request_messages import UpdateRequestMessage

# grid relative
from ...auth import error_handler
from ...auth import optional_token
from ...auth import token_required
from ..blueprint import dcfl_blueprint as dcfl_route


@dcfl_route.route("/requests", methods=["POST"])
@token_required
def create_request(current_user):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}

    status_code, response_msg = error_handler(
        route_logic, 200, CreateRequestMessage, current_user, content
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
        route_logic, 200, GetRequestMessage, current_user, content
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
        route_logic, 200, GetRequestsMessage, current_user, content
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
        route_logic, 200, UpdateRequestMessage, current_user, content
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
        route_logic, 204, DeleteRequestMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        dumps(response),
        status=status_code,
        mimetype="application/json",
    )
