from .blueprint import users_blueprint as user_route
from flask import request, Response
import json
from ...core.node import node
from nacl.encoding import HexEncoder


@user_route.route("/", methods=["POST"])
def create_user():
    mock_response = {"msg": "User created succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@user_route.route("/", methods=["GET"])
def get_all_users():
    mock_response = {"users": ["Bob", "Alice", "James"]}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@user_route.route("/<user_id>", methods=["GET"])
def get_specific_user(user_id):
    mock_response = {"user": {"name": "Bob", "id": user_id}}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@user_route.route("/search", methods=["POST"])
def search_users():
    mock_response = {"users": ["Bob", "Alice", "James"]}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@user_route.route("/<user_id>/email", methods=["PUT"])
def change_email(user_id):
    mock_response = {"msg": "User email was changed succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@user_route.route("/<user_id>/password", methods=["PUT"])
def change_password(user_id):
    mock_response = {"msg": "User password was changed succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@user_route.route("/<user_id>/role", methods=["PUT"])
def change_role(user_id):
    mock_response = {"msg": "User role was changed succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@user_route.route("/<user_id>", methods=["DELETE"])
def delete_user(user_id):
    mock_response = {"msg": "User was deleted succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@user_route.route("/login", methods=["POST"])
def user_login():
    mock_response = {
        "msg": "Successfully logged in!",
        "key": node.signing_key.encode(encoder=HexEncoder).decode("utf-8"),
        "metadata": node.get_metadata_for_client(),
    }
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )
