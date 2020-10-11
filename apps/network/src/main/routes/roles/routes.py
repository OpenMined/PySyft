from .blueprint import roles_blueprint as roles_route
from flask import request, Response
import json


@roles_route.route("/", methods=["POST"])
def create_role():
    mock_response = {"msg": "Role created succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@roles_route.route("/", methods=["GET"])
def get_all_roles():
    mock_response = {"roles": ["RoleA", "RoleB", "RoleC"]}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@roles_route.route("/<role_id>", methods=["GET"])
def get_specific_role(role_id):
    mock_response = {"role": {"name": "Role A", "id": role_id}}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@roles_route.route("/<role_id>", methods=["PUT"])
def update_role(role_id):
    mock_response = {"msg": "Role was updated succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@roles_route.route("/<role_id>", methods=["DELETE"])
def delete_role(role_id):
    mock_response = {"msg": "Role was deleted succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )
