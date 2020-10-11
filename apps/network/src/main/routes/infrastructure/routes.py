from .blueprint import networks_blueprint as network_route
from flask import request, Response
import json


@network_route.route("/", methods=["POST"])
def create_network():
    mock_response = {"msg": "Network created succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@network_route.route("/", methods=["GET"])
def get_all_networks():
    mock_response = {"networks": ["Net-Gama", "Net-Beta", "Net-Pi"]}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@network_route.route("/<network_id>", methods=["GET"])
def get_specific_network(network_id):
    mock_response = {"network": {"name": "Net-Gama", "id": network_id}}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@network_route.route("/<network_id>", methods=["PUT"])
def update_network(network_id):
    mock_response = {"msg": "Network was updated succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@network_route.route("/<network_id>", methods=["DELETE"])
def delete_network(network_id):
    mock_response = {"msg": "Network was deleted succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@network_route.route("/autoscaling", methods=["POST"])
def create_autoscaling():
    mock_response = {"msg": "Network auto-scaling created succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@network_route.route("/autoscaling/", methods=["GET"])
def get_all_autoscalings():
    mock_response = {"auto-scalings": ["Condition 1", "Condition 2", "Condition 3"]}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@network_route.route("/autoscaling/<network_id>", methods=["GET"])
def get_specific_autoscaling(network_id):
    mock_response = {"network": {"name": "Net-Gama", "id": network_id}}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@network_route.route("/autoscaling/<network_id>", methods=["PUT"])
def update_autoscaling(network_id):
    mock_response = {"msg": "Network auto-scaling was updated succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )


@network_route.route("/autoscaling/<network_id>", methods=["DELETE"])
def delete_autoscaling(network_id):
    mock_response = {"msg": "Network auto-scaling was deleted succesfully!"}
    return Response(
        status=200, response=json.dumps(mock_response), mimetype="application/json"
    )
