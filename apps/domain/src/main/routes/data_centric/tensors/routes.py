from ..blueprint import dcfl_blueprint as dcfl_route
from flask import request, Response
import json


@dcfl_route.route("/tensors", methods=["POST"])
def create_tensor():
    mock_response = {"msg": "tensor created succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@dcfl_route.route("/tensors/<tensor_id>", methods=["GET"])
def get_tensor(tensor_id):
    mock_response = {
        "tensor": {
            "id": tensor_id,
            "tags": ["tensor-a"],
            "description": "tensor sample",
        }
    }
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@dcfl_route.route("/tensors", methods=["GET"])
def get_all_tensors():
    mock_response = {
        "tensors": [
            {
                "id": "35654sad6ada",
                "tags": ["tensor-a"],
                "description": "tensor sample",
            },
            {
                "id": "adfarf3f1af5",
                "tags": ["tensor-b"],
                "description": "tensor sample",
            },
            {"id": "fas4e6e1fas", "tags": ["tensor-c"], "description": "tensor sample"},
        ]
    }
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@dcfl_route.route("/tensors/<tensor_id>", methods=["PUT"])
def update_tensor(tensor_id):
    mock_response = {"msg": "tensor changed succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@dcfl_route.route("/tensors/<tensor_id>", methods=["DELETE"])
def delete_tensor(tensor_id):
    mock_response = {"msg": "tensor deleted succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )
