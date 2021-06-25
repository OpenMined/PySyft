# stdlib
import json

# third party
from flask import Response
from flask import request
from syft.grid.messages.tensor_messages import CreateTensorMessage
from syft.grid.messages.tensor_messages import DeleteTensorMessage
from syft.grid.messages.tensor_messages import GetTensorMessage
from syft.grid.messages.tensor_messages import GetTensorsMessage
from syft.grid.messages.tensor_messages import UpdateTensorMessage

# grid relative
from ....core.task_handler import route_logic
from ...auth import error_handler
from ...auth import token_required
from ..blueprint import dcfl_blueprint as dcfl_route


@dcfl_route.route("/tensors", methods=["POST"])
@token_required
def create_tensor(current_user):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}

    status_code, response_msg = error_handler(
        route_logic, 201, CreateTensorMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@dcfl_route.route("/tensors/<tensor_id>", methods=["GET"])
@token_required
def get_tensor(current_user, tensor_id):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}

    content["tensor_id"] = tensor_id

    status_code, response_msg = error_handler(
        route_logic, 200, GetTensorMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@dcfl_route.route("/tensors", methods=["GET"])
def get_all_tensors():
    # Get request body
    content = request.get_json()
    if not content:
        content = {}

    status_code, response_msg = error_handler(
        route_logic, 200, GetTensorsMessage, None, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@dcfl_route.route("/tensors/<tensor_id>", methods=["PUT"])
@token_required
def update_tensor(current_user, tensor_id):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}

    content["tensor_id"] = tensor_id

    status_code, response_msg = error_handler(
        route_logic, 204, UpdateTensorMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@dcfl_route.route("/tensors/<tensor_id>", methods=["DELETE"])
@token_required
def delete_tensor(current_user, tensor_id):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}

    content["tensor_id"] = tensor_id

    status_code, response_msg = error_handler(
        route_logic, 204, DeleteTensorMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        json.dumps(response),
        status=status_code,
        mimetype="application/json",
    )
