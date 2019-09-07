"""
    All Gateway routes (REST API).
"""

from flask import render_template, Response, request, current_app
from . import main
import json
import random
import os
import requests


# All grid nodes registered at grid network will be stored here
grid_nodes = {}


@main.route("/", methods=["GET"])
def index():
    """ Main Page. """
    return render_template("index.html")


@main.route("/join", methods=["POST"])
def join_grid_node():
    """ Register a new grid node at grid network. 
        TODO: Add Authentication process.
    """
    data = json.loads(request.data)

    # Invalid body
    if "node-id" not in data or "node-address" not in data:
        return Response("", status=400, mimetype="application/json")
    # Grid ID already registered
    if data["node-id"] in grid_nodes:
        return Response(
            "This ID has already been registered",
            status=403,
            mimetype="appication/json",
        )

    # Add new grid node to list of known nodes
    grid_nodes[data["node-id"]] = data["node-address"]
    return Response("Successfully Connected!", status=200, mimetype="application/json")


@main.route("/connected-nodes", methods=["GET"])
def get_connected_nodes():
    """ Get a list of connected nodes. """
    return Response(
        json.dumps({"grid-nodes": list(grid_nodes.keys())}),
        status=200,
        mimetype="application/json",
    )


@main.route("/choose-model-host", methods=["GET"])
def choose_model_host():
    """ Used to choose some grid node to host a model.
        PS: Currently we perform this randomly.
    """
    n_replica = current_app.config["N_REPLICA"]
    if not n_replica:
        n_replica = 1

    hosts = random.sample(list(grid_nodes.keys()), n_replica)
    hosts_info = [(host, grid_nodes[host]) for host in hosts]
    return Response(json.dumps(hosts_info), status=200, mimetype="application/json")


@main.route("/search-model", methods=["POST"])
def search_model():
    """ Search for a plain text model on the grid network. """
    body = json.loads(request.data)

    # Invalid body
    if "model_id" not in body:
        return Response("", status=400, mimetype="application/json")

    match_nodes = []
    for node in grid_nodes:
        try:
            response = requests.get(grid_nodes[node] + "/models/").content
        except ConnectionError:
            continue
        response = json.loads(response)
        if body["model_id"] in response.get("models", []):
            match_nodes.append((node, grid_nodes[node]))

    # Return a list[ (id, address) ]  with all grid nodes that have the desired model
    return Response(json.dumps(match_nodes), status=200, mimetype="application/json")


@main.route("/search-available-models", methods=["GET"])
def available_models():
    """ Get all available models on the grid network. Can be useful to know what models our grid network have. """
    models = set()
    for node in grid_nodes:
        try:
            response = requests.get(grid_nodes[node] + "/models/").content
        except ConnectionError:
            continue
        response = json.loads(response)
        models.update(set(response.get("models", [])))

    # Return a list[ "model_id" ]  with all grid nodes
    return Response(json.dumps(list(models)), status=200, mimetype="application/json")


@main.route("/search-available-tags", methods=["GET"])
def available_tags():
    """ Returns all available tags stored on grid nodes. Can be useful to know what dataset our grid network have. """
    tags = set()
    for node in grid_nodes:
        try:
            response = requests.get(grid_nodes[node] + "/dataset-tags").content
        except ConnectionError:
            continue
        response = json.loads(response)
        tags.update(set(response))

    # Return a list[ "#tags" ]  with all grid nodes
    return Response(json.dumps(list(tags)), status=200, mimetype="application/json")


@main.route("/search", methods=["POST"])
def search_dataset_tags():
    """ Search for information on all known nodes and return a list of the nodes that own it. """
    body = json.loads(request.data)

    # Invalid body
    if "query" not in body:
        return Response("", status=400, mimetype="application/json")

    # Perform requests (HTTP) to all known nodes looking for the desired data tag
    match_grid_nodes = []
    for node in grid_nodes:
        try:
            response = requests.post(
                grid_nodes[node] + "/search", data=json.dumps({"query": body["query"]})
            ).content
        except ConnectionError:
            continue
        response = json.loads(response)
        # If contains
        if response["content"]:
            match_grid_nodes.append((node, grid_nodes[node]))

    # Return a list[ (id, address) ]  with all grid nodes that have the desired data
    return Response(
        json.dumps(match_grid_nodes), status=200, mimetype="application/json"
    )
