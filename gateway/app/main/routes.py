"""
    All Gateway routes (REST API).
"""

from flask import render_template, Response, request, current_app
from . import main
import json
import random
import os
import requests

from .persistence.manager import register_new_node, connected_nodes


# All grid nodes registered at grid network will be stored here
grid_nodes = {}

SMPC_HOST_CHUNK = 4  # Minimum nodes required to host an encrypted model


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
        return Response(
            {"message": "Invalid json."}, status=400, mimetype="application/json"
        )

    # Register new node
    if register_new_node(data["node-id"], data["node-address"]):
        return Response(
            {"message": "Successfully Connected!"},
            status=200,
            mimetype="application/json",
        )

    else:  # Grid ID already registered
        return Response(
            {"message": "This ID has already been registered"},
            status=409,
            mimetype="appication/json",
        )


@main.route("/connected-nodes", methods=["GET"])
def get_connected_nodes():
    """ Get a list of connected nodes. """
    grid_nodes = connected_nodes()
    return Response(
        json.dumps({"grid-nodes": list(grid_nodes.keys())}),
        status=200,
        mimetype="application/json",
    )


@main.route("/choose-encrypted-model-host", methods=["GET"])
def choose_encrypted_model_host():
    """ Used to choose grid nodes to host an encrypted model
        PS: currently we perform this randomly
    """
    grid_nodes = connected_nodes()
    n_replica = current_app.config["N_REPLICA"]
    if not n_replica:
        n_replica = 1
    try:
        hosts = random.sample(list(grid_nodes.keys()), n_replica * SMPC_HOST_CHUNK)
        hosts_info = [(host, grid_nodes[host]) for host in hosts]
    # If grid network doesn't have enough grid nodes
    except ValueError:
        hosts_info = []
    return Response(json.dumps(hosts_info), status=200, mimetype="application/json")


@main.route("/choose-model-host", methods=["GET"])
def choose_model_host():
    """ Used to choose some grid node to host a model.
        PS: Currently we perform this randomly.
    """
    grid_nodes = connected_nodes()
    n_replica = current_app.config["N_REPLICA"]
    if not n_replica:
        n_replica = 1

    hosts = random.sample(list(grid_nodes.keys()), n_replica)
    hosts_info = [(host, grid_nodes[host]) for host in hosts]
    return Response(json.dumps(hosts_info), status=200, mimetype="application/json")


@main.route("/search-encrypted-model", methods=["POST"])
def search_encrypted_model():
    """ Search for an encrypted plan model on the grid network, if found,
        returns host id, host address and SMPC workers infos.
    """
    body = json.loads(request.data)

    if "model_id" not in body:
        return Response(
            {"message": "Invalid json fields."}, status=400, mimetype="application/json"
        )

    grid_nodes = connected_nodes()
    match_nodes = {}
    for node in grid_nodes:
        try:
            response = requests.post(
                os.path.join(grid_nodes[node], "search-encrypted-models"),
                data=request.data,
            )
        except requests.exceptions.ConnectionError:
            continue

        response = json.loads(response.content)
        # If workers / crypto_provider fields in response dict
        if not len({"workers", "crypto_provider"} - set(response.keys())):
            match_nodes[node] = {"address": grid_nodes[node], "nodes": response}
    return Response(json.dumps(match_nodes), status=200, mimetype="application/json")


@main.route("/search-model", methods=["POST"])
def search_model():
    """ Search for a plain text model on the grid network. """
    body = json.loads(request.data)

    # Invalid body
    if "model_id" not in body:
        return Response(
            {"message": "Invalid json fields."}, status=400, mimetype="application/json"
        )

    grid_nodes = connected_nodes()
    match_nodes = []
    for node in grid_nodes:
        try:
            response = requests.get(grid_nodes[node] + "/models/").content
        except requests.exceptions.ConnectionError:
            continue
        response = json.loads(response)
        if body["model_id"] in response.get("models", []):
            match_nodes.append((node, grid_nodes[node]))

    # Return a list[ (id, address) ]  with all grid nodes that have the desired model
    return Response(json.dumps(match_nodes), status=200, mimetype="application/json")


@main.route("/search-available-models", methods=["GET"])
def available_models():
    """ Get all available models on the grid network. Can be useful to know what models our grid network have. """
    grid_nodes = connected_nodes()
    models = set()
    for node in grid_nodes:
        try:
            response = requests.get(grid_nodes[node] + "/models/").content
        except requests.exceptions.ConnectionError:
            continue
        response = json.loads(response)
        models.update(set(response.get("models", [])))

    # Return a list[ "model_id" ]  with all grid nodes
    return Response(json.dumps(list(models)), status=200, mimetype="application/json")


@main.route("/search-available-tags", methods=["GET"])
def available_tags():
    """ Returns all available tags stored on grid nodes. Can be useful to know what dataset our grid network have. """
    grid_nodes = connected_nodes()
    tags = set()
    for node in grid_nodes:
        try:
            response = requests.get(grid_nodes[node] + "/dataset-tags").content
        except requests.exceptions.ConnectionError:
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
        return Response(
            {"message": "Invalid json fields."}, status=400, mimetype="application/json"
        )

    grid_nodes = connected_nodes()
    # Perform requests (HTTP) to all known nodes looking for the desired data tag
    match_grid_nodes = []
    for node in grid_nodes:
        try:
            response = requests.post(
                grid_nodes[node] + "/search", data=json.dumps({"query": body["query"]})
            ).content
        except requests.exceptions.ConnectionError:
            continue
        response = json.loads(response)
        # If contains
        if response["content"]:
            match_grid_nodes.append((node, grid_nodes[node]))

    # Return a list[ (id, address) ]  with all grid nodes that have the desired data
    return Response(
        json.dumps(match_grid_nodes), status=200, mimetype="application/json"
    )
