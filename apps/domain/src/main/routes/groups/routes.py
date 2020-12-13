from flask import request, Response
from json import loads, dumps

import json

from syft.core.node.common.service.repr_service import ReprMessage

from ...core.task_handler import task_handler, process_as_syft_message
from .blueprint import groups_blueprint as group_route
from ..auth import error_handler, token_required
from ...core.node import node
from ...core.groups.group_ops import (
    create_group,
    get_group,
    get_all_groups,
    put_group,
    delete_group,
)
from ...core.exceptions import (
    AuthorizationError,
    GroupNotFoundError,
    InvalidCredentialsError,
    MissingRequestKeyError,
    PyGridError,
    RoleNotFoundError,
    UserNotFoundError,
)


@group_route.route("", methods=["POST"])
@token_required
def create_group_route(current_user):
    def route_logic(current_user):
        # Get request body
        content = loads(request.data)
        content["current_user"] = current_user

        # Execute task
        response_body = task_handler(
            route_function=create_group,
            data=content,
            mandatory={
                "current_user": MissingRequestKeyError,
                "name": MissingRequestKeyError,
            },
        )
        return response_body

    status_code, response_body = error_handler(route_logic, current_user)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@group_route.route("", methods=["GET"])
@token_required
def get_all_groups_routes(current_user):
    def route_logic(current_user):
        # Execute task
        response_body = task_handler(
            route_function=get_all_groups,
            data={"current_user": current_user},
            mandatory={
                "current_user": MissingRequestKeyError,
            },
        )
        return response_body

    status_code, response_body = error_handler(route_logic, current_user)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@group_route.route("/<group_id>", methods=["GET"])
@token_required
def get_specific_group_route(current_user, group_id):
    def route_logic(current_user, group_id):

        # Execute task
        response_body = task_handler(
            route_function=get_group,
            data={"group_id": group_id, "current_user": current_user},
            mandatory={
                "group_id": MissingRequestKeyError,
                "current_user": MissingRequestKeyError,
            },
        )
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, group_id)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@group_route.route("/<group_id>", methods=["PUT"])
@token_required
def update_group_route(current_user, group_id):
    def route_logic(current_user, group_id):
        # Get request body
        content = loads(request.data)
        content = {"new_fields": content}
        content["current_user"] = current_user
        content["group_id"] = group_id

        # Execute task
        response_body = task_handler(
            route_function=put_group,
            data=content,
            mandatory={
                "current_user": MissingRequestKeyError,
                "group_id": MissingRequestKeyError,
                "new_fields": MissingRequestKeyError,
            },
        )
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, group_id)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )


@group_route.route("/<group_id>", methods=["DELETE"])
@token_required
def delete_group_route(current_user, group_id):
    def route_logic(current_user, group_id):
        # Execute task
        response_body = task_handler(
            route_function=delete_group,
            data={"group_id": group_id, "current_user": current_user},
            mandatory={
                "current_user": MissingRequestKeyError,
                "group_id": MissingRequestKeyError,
            },
        )
        return response_body

    status_code, response_body = error_handler(route_logic, current_user, group_id)

    return Response(
        dumps(response_body), status=status_code, mimetype="application/json"
    )
