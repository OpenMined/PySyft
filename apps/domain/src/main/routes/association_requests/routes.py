from .blueprint import association_requests_blueprint as association_request_route
from flask import request, Response
import json


@association_request_route.route("/request", methods=["POST"])
def send_association_request():
    mock_response = {"msg": "Association request sent!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@association_request_route.route("/receive", methods=["POST"])
def recv_association_request():
    mock_response = {"msg": "Association request received!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@association_request_route.route("/respond", methods=["POST"])
def reply_association_request():
    mock_response = {"msg": "Association request was replied!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@association_request_route.route("/", methods=["GET"])
def get_all_association_requests():
    mock_response = {"association-requests": ["Network A", "Network B", "Network C"]}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@association_request_route.route("/<association_request_id>", methods=["GET"])
def get_specific_association_requests(association_request_id):
    mock_response = {
        "association-request": {
            "ID": association_request_id,
            "address": "156.89.33.200",
        }
    }
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@association_request_route.route("/<association_request_id>", methods=["DELETE"])
def delete_association_requests(association_request_id):
    mock_response = {"msg": "Association request deleted!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )
