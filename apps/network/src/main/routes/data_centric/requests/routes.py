from ..blueprint import dcfl_blueprint as dcfl_route
from flask import request, Response
import json


@dcfl_route.route("/requests", methods=["POST"])
def create_request():
    mock_response = {"msg": "Request created succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@dcfl_route.route("/requests/<request_id>", methods=["GET"])
def get_specific_request(request_id):
    mock_response = {"request": {"id": request_id, "reason": "request reason"}}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@dcfl_route.route("/requests", methods=["GET"])
def get_all_requests():
    mock_response = {
        "requests": [
            {"id": "35654sad6ada", "reason": "request A reason"},
            {"id": "adfarf3f1af5", "reason": "request B reason"},
            {"id": "fas4e6e1fas", "reason": "request C reason"},
        ]
    }
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@dcfl_route.route("/requests/<request_id>", methods=["PUT"])
def update_request(request_id):
    mock_response = {"msg": "Request updated succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@dcfl_route.route("/requests/<request_id>", methods=["DELETE"])
def delete_request(request_id):
    mock_response = {"msg": "Request deleted succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )
