from ..blueprint import dcfl_blueprint as dcfl_route
from flask import request, Response
import json

## Nodes CRUD
@dcfl_route.route("/nodes", methods=["POST"])
def create_node():
    mock_response = {"msg": "Node created succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@dcfl_route.route("/nodes/<node_id>", methods=["GET"])
def get_node(node_id):
    mock_json = {
        "node": {"id": node_id, "tags": ["node-a"], "description": "node sample"}
    }
    return Response(
        status=200, response=json.dumps(mock_json), mimetype="application/json"
    )


@dcfl_route.route("/nodes", methods=["GET"])
def get_all_nodes():
    mock_json = {
        "nodes": [
            {"id": "35654sad6ada", "address": "175.89.0.170"},
            {"id": "adfarf3f1af5", "address": "175.55.22.150"},
            {"id": "fas4e6e1fas", "address": "195.74.128.132"},
        ]
    }
    return Response(
        status=200, response=json.dumps(mock_json), mimetype="application/json"
    )


@dcfl_route.route("/nodes/<node_id>", methods=["PUT"])
def update_node(node_id):
    mock_response = {"msg": "Node changed succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@dcfl_route.route("/nodes/<node_id>", methods=["DELETE"])
def delete_node(node_id):
    mock_response = {"msg": "Node deleted succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


## Auto-scaling CRUD


@dcfl_route.route("/nodes/autoscaling", methods=["POST"])
def create_autoscaling():
    mock_response = {"msg": "Autoscaling condition created succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@dcfl_route.route("/nodes/autoscaling", methods=["GET"])
def get_all_autoscaling_conditions():
    mock_response = {
        "condition_a": {"mem_usage": "80%", "cpu_usage": "90%", "disk_usage": "75%"},
        "condition_b": {"mem_usage": "50%", "cpu_usage": "70%", "disk_usage": "95%"},
        "condition_c": {"mem_usage": "92%", "cpu_usage": "77%", "disk_usage": "50%"},
    }
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@dcfl_route.route("/nodes/autoscaling/<autoscaling_id>", methods=["GET"])
def get_specific_autoscaling_condition(autoscaling_id):
    mock_response = {"mem_usage": "80%", "cpu_usage": "90%", "disk_usage": "75%"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@dcfl_route.route("/nodes/autoscaling/<autoscaling_id>", methods=["PUT"])
def update_autoscaling_condition(autoscaling_id):
    mock_response = {"msg": "Autoscaling condition updated succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@dcfl_route.route("/nodes/autoscaling/<autoscaling_id>", methods=["DELETE"])
def delete_autoscaling_condition(autoscaling_id):
    mock_response = {"msg": "Autoscaling condition deleted succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


## Workers CRUD


@dcfl_route.route("/workers", methods=["POST"])
def create_worker():
    mock_response = {"msg": "Worker created succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@dcfl_route.route("/workers", methods=["GET"])
def get_all_workers():
    mock_response = {
        "workers": [
            {"id": "546513231a", "address": "159.156.128.165", "datasets": 25320},
            {"id": "asfa16f5aa", "address": "138.142.125.125", "datasets": 2530},
            {"id": "af61ea3a3f", "address": "19.16.98.146", "datasets": 2320},
            {"id": "af4a51adas", "address": "15.59.18.165", "datasets": 5320},
        ]
    }

    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@dcfl_route.route("/workers/<worker_id>", methods=["GET"])
def get_specific_worker(worker_id):
    mock_response = {
        "worker": {"id": worker_id, "address": "159.156.128.165", "datasets": 25320}
    }
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@dcfl_route.route("/workers/<worker_id>", methods=["DELETE"])
def delete_worker(worker_id):
    mock_response = {"msg": "Worker was deleted succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )
