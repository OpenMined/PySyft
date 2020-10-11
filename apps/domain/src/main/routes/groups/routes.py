from .blueprint import groups_blueprint as group_route
from flask import request, Response
import json


@group_route.route("/", methods=["POST"])
def create_group():
    mock_response = {"msg": "Group created succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@group_route.route("/", methods=["GET"])
def get_all_groups():
    mock_response = {"groups": ["GroupA", "GroupB", "GroupC"]}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@group_route.route("/<group_id>", methods=["GET"])
def get_specific_group(group_id):
    mock_response = {"group": {"name": "Group A", "id": group_id}}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@group_route.route("/<group_id>", methods=["PUT"])
def update_group(group_id):
    mock_response = {"msg": "Group was updated succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@group_route.route("/<group_id>", methods=["DELETE"])
def delete_group(group_id):
    mock_response = {"msg": "Group was deleted succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )
