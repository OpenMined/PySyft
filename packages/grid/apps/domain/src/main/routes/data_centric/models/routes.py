# stdlib
from json import dumps

# third party
from flask import Response
from flask import request
from main.core.task_handler import route_logic
from syft.grid.messages.model_messages import DeleteModelMessage
from syft.grid.messages.model_messages import GetModelMessage
from syft.grid.messages.model_messages import GetModelsMessage
from syft.grid.messages.model_messages import UpdateModelMessage

# grid relative
from ...auth import error_handler
from ...auth import token_required
from ..blueprint import dcfl_blueprint as dcfl_route


@dcfl_route.route("/models/<model_id>", methods=["GET"])
@token_required
def get_model_info(current_user, model_id):
    content = {}
    content["current_user"] = current_user
    content["model_id"] = model_id
    status_code, response_msg = error_handler(
        route_logic, 200, GetModelMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@dcfl_route.route("/models", methods=["GET"])
@token_required
def get_all_models_info(current_user):
    content = {}
    content["current_user"] = current_user

    status_code, response_msg = error_handler(
        route_logic, 200, GetModelsMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@dcfl_route.route("/models/<model_id>", methods=["PUT"])
@token_required
def update_model(current_user, model_id):
    # Get request body
    content = request.get_json()
    content["current_user"] = current_user
    content["model_id"] = model_id
    status_code, response_msg = error_handler(
        route_logic, 204, UpdateModelMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        dumps(response),
        status=status_code,
        mimetype="application/json",
    )


@dcfl_route.route("/models/<model_id>", methods=["DELETE"])
@token_required
def delete_model(current_user, model_id):
    # Get request body
    content = {}
    content["current_user"] = current_user
    content["model_id"] = model_id

    status_code, response_msg = error_handler(
        route_logic, 204, DeleteModelMessage, current_user, content
    )

    response = response_msg if isinstance(response_msg, dict) else response_msg.content

    return Response(
        dumps(response),
        status=status_code,
        mimetype="application/json",
    )
