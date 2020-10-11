from ..blueprint import dcfl_blueprint as dcfl_route
from flask import request, Response
import json


@dcfl_route.route("/datasets", methods=["POST"])
def create_dataset():
    mock_response = {"msg": "Dataset created succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@dcfl_route.route("/datasets/<dataset_id>", methods=["GET"])
def get_dataset(dataset_id):
    mock_response = {
        "dataset": {
            "id": dataset_id,
            "tags": ["dataset-a"],
            "description": "Dataset sample",
        }
    }
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@dcfl_route.route("/datasets", methods=["GET"])
def get_all_datasets():
    mock_response = {
        "datasets": [
            {
                "id": "35654sad6ada",
                "tags": ["dataset-a"],
                "description": "Dataset sample",
            },
            {
                "id": "adfarf3f1af5",
                "tags": ["dataset-b"],
                "description": "Dataset sample",
            },
            {
                "id": "fas4e6e1fas",
                "tags": ["dataset-c"],
                "description": "Dataset sample",
            },
        ]
    }
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@dcfl_route.route("/datasets/<dataset_id>", methods=["PUT"])
def update_dataset(dataset_id):
    mock_response = {"msg": "Dataset changed succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@dcfl_route.route("/datasets/<dataset_id>", methods=["DELETE"])
def delete_dataset(dataset_id):
    mock_response = {"msg": "Dataset deleted succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )
