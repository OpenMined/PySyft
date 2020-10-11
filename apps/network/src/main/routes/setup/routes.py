from .blueprint import setup_blueprint as setup_route
from flask import request, Response
import json


@setup_route.route("/", methods=["POST"])
def initial_setup():
    mock_response = {"msg": "Running initial setup!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@setup_route.route("/", methods=["GET"])
def get_setup():
    mock_response = {"setup": {}}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )
