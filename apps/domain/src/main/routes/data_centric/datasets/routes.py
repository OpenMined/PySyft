from json import dumps, loads

from flask import request, Response
from syft.core.node.common.service.repr_service import ReprMessage
from syft.grid.messages.dataset_messages import CreateDatasetMessage
from syft.grid.messages.dataset_messages import (
    CreateDatasetMessage,
    GetDatasetMessage,
    GetDatasetsMessage,
    UpdateDatasetMessage,
    DeleteDatasetMessage,
)

from ...auth import error_handler, token_required, optional_token
from main.core.task_handler import route_logic, task_handler
from ..blueprint import dcfl_blueprint as dcfl_route
from ....core.node import node


@dcfl_route.route("/datasets", methods=["POST"])
@token_required
def create_dataset(current_user):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}
    content["current_user"] = current_user
    status_code, response_msg = error_handler(
        route_logic, CreateDatasetMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@dcfl_route.route("/datasets/<dataset_id>", methods=["GET"])
@token_required
def get_dataset(current_user, dataset_id):
    content = {}
    content["current_user"] = current_user
    content["dataset_id"] = dataset_id
    status_code, response_msg = error_handler(
        route_logic, GetDatasetMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@dcfl_route.route("/datasets", methods=["GET"])
@token_required
def get_all_datasets(current_user):
    content = {}
    content["current_user"] = current_user
    status_code, response_msg = error_handler(
        route_logic, GetDatasetsMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@dcfl_route.route("/datasets/<dataset_id>", methods=["PUT"])
@token_required
def update_dataset(current_user, dataset_id):
    # Get request body
    content = request.get_json()
    if not content:
        content = {}
    content["current_user"] = current_user
    content["dataset_id"] = dataset_id
    status_code, response_msg = error_handler(
        route_logic, UpdateDatasetMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content
    if status_code == 200:
        status_code = 204

    return Response(
        dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@dcfl_route.route("/datasets/<dataset_id>", methods=["DELETE"])
@token_required
def delete_dataset(current_user, dataset_id):
    # Get request body
    content = {}
    content["current_user"] = current_user
    content["dataset_id"] = dataset_id
    status_code, response_msg = error_handler(
        route_logic, DeleteDatasetMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content
    if status_code == 200:
        status_code = 204

    return Response(
        dumps(response),
        status=status_code,
        mimetype="application/json",
    )
