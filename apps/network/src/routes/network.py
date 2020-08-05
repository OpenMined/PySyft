# PyGrid imports
import json

# General imports
import os
import random

import requests
from flask import Response, current_app, request

from .. import http
from ..network import network_manager

# Store all grid nodes that are registered in the grid network
grid_nodes = {}
SMPC_HOST_CHUNK = 4  # Minimum nodes required to host an encrypted model
INVALID_JSON_FORMAT_MESSAGE = (
    "Invalid JSON format."  # Default message used to report Invalid JSON format.
)


@http.route("/join", methods=["POST"])
def join_grid_node():
    """Register a new grid node at grid network.

    TODO: Add Authentication process.
    """

    response_body = {"message": None}
    status_code = None

    try:
        data = json.loads(request.data)
        # Register new node
        if network_manager.register_new_node(data["node-id"], data["node-address"]):
            response_body["message"] = "Successfully Connected!"
            status_code = 200
        else:  # Grid ID already registered
            response_body["message"] = "This ID has already been registered"
            status_code = 409

    # JSON format not valid.
    except (ValueError, KeyError) as e:
        response_body["message"] = INVALID_JSON_FORMAT_MESSAGE
        status_code = 400
    except Exception as e:
        response_body["message"] = str(e)
        status_code = 500  # Internal Server Error

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


@http.route("/connected-nodes", methods=["GET"])
def get_connected_nodes():
    """Return a response object containing a list of all the connected
    nodes."""
    grid_nodes = network_manager.connected_nodes()
    return Response(
        json.dumps({"grid-nodes": list(grid_nodes.keys())}),
        status=200,
        mimetype="application/json",
    )


@http.route("/delete-node", methods=["DELETE"])
def delete_grid_node():
    """Delete a grid node in the grid network."""

    response_body = {"message": None}
    status_code = None

    try:
        data = json.loads(request.data)

        # Register new node
        if network_manager.delete_node(data["node-id"], data["node-address"]):
            response_body["message"] = "Successfully Deleted!"
            status_code = 200
        else:  # Grid ID was not found
            response_body["message"] = "This ID was not found in connected nodes"
            status_code = 409

    # JSON format not valid.
    except (ValueError, KeyError) as e:
        response_body["message"] = INVALID_JSON_FORMAT_MESSAGE
        status_code = 400
    except Exception as e:
        response_body["message"] = str(e)
        status_code = 500  # Internal Server Error

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


@http.route("/choose-encrypted-model-host", methods=["GET"])
def choose_encrypted_model_host():
    """Choose grid nodes to host an encrypted model (currently the choice is
    random)."""

    hosts_info = []
    response_body = {"message": None}
    status_code = None

    if "N_REPLICA" in current_app.config.values():
        n_replica = current_app.config["N_REPLICA"]
    else:
        n_replica = 1

    try:
        grid_nodes = network_manager.connected_nodes()

        hosts = random.sample(list(grid_nodes.keys()), n_replica * SMPC_HOST_CHUNK)
        hosts_info = [(host, grid_nodes[host]) for host in hosts]

        response_body = hosts_info
        status_code = 200

    # If grid network doesn't have enough grid nodes
    except ValueError:
        response_body = hosts_info
        status_code = 400
    except Exception as e:
        response_body["message"] = str(e)
        status_code = 500  # Internal Server Error

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


@http.route("/choose-model-host", methods=["GET"])
def choose_model_host():
    """Choose n grid nodes to host a model (currently the choice is random)."""
    grid_nodes = network_manager.connected_nodes()
    n_replica = current_app.config["N_REPLICA"]
    if not n_replica:
        n_replica = 1

    model_id = request.args.get("model_id")
    hosts_info = None

    # Lookup the nodes already hosting this model to prevent hosting different model versions
    if model_id:
        hosts_info = _get_model_hosting_nodes(model_id)

    # No model id given or no hosting nodes found: randomly choose node
    if not hosts_info:
        hosts = random.sample(list(grid_nodes.keys()), n_replica)
        hosts_info = [(host, grid_nodes[host]) for host in hosts]

    return Response(json.dumps(hosts_info), status=200, mimetype="application/json")


@http.route("/search-encrypted-model", methods=["POST"])
def search_encrypted_model():
    """Search for an encrypted plan model in the grid network and, if found,
    return host id, host address, and SMPC workers information."""

    response_body = {"message": None}
    status_code = None

    try:
        grid_nodes = network_manager.connected_nodes()
        match_nodes = {}
        for node in grid_nodes:
            try:
                response = requests.post(
                    os.path.join(
                        grid_nodes[node], "/data-centric/search-encrypted-models"
                    ),
                    data=request.data,
                )
            except requests.exceptions.ConnectionError:
                continue

            response = json.loads(response.content)

            # If workers / crypto_provider fields in response dict
            if not len({"workers", "crypto_provider"} - set(response.keys())):
                match_nodes[node] = {"address": grid_nodes[node], "nodes": response}

            response_body = match_nodes
            status_code = 200

    # JSON format not valid.
    except (ValueError, KeyError) as e:
        response_body["message"] = INVALID_JSON_FORMAT_MESSAGE
        status_code = 400
    except Exception as e:
        response_body["message"] = str(e)
        status_code = 500  # Internal Server Error

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


@http.route("/search-model", methods=["POST"])
def search_model():
    """Search for a plain text model in the grid network."""

    response_body = {"message": None}
    status_code = None

    try:
        body = json.loads(request.data)

        model_id = body["model_id"]
        match_nodes = _get_model_hosting_nodes(model_id)

        # Return a list[(id, address)]  with all grid nodes that have the desired model
        response_body = match_nodes
        status_code = 200

    except (ValueError, KeyError):
        response_body["message"] = INVALID_JSON_FORMAT_MESSAGE
        status_code = 400
    except Exception as e:
        response_body["message"] = str(e)
        status_code = 500  # Internal Server Error

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


@http.route("/search-available-models", methods=["GET"])
def available_models():
    """Get all available models in the grid network."""
    grid_nodes = network_manager.connected_nodes()
    models = set()
    for node in grid_nodes:
        try:
            response = requests.get(grid_nodes[node] + "/data-centric/models/").content
        except requests.exceptions.ConnectionError:
            continue
        response = json.loads(response)
        models.update(set(response.get("models", [])))

    # Return a list["model_id"]  with all grid nodes
    return Response(json.dumps(list(models)), status=200, mimetype="application/json")


@http.route("/search-available-tags", methods=["GET"])
def available_tags():
    """Return all available dataset tags stored on grid nodes."""
    grid_nodes = network_manager.connected_nodes()
    tags = set()
    for node in grid_nodes:
        try:
            response = requests.get(
                grid_nodes[node] + "/data-centric/dataset-tags"
            ).content
        except requests.exceptions.ConnectionError:
            continue
        response = json.loads(response)
        tags.update(set(response))

    # Return a list["#tags"]  with all grid nodes
    return Response(json.dumps(list(tags)), status=200, mimetype="application/json")


@http.route("/search", methods=["POST"])
def search_dataset_tags():
    """Search for information on all known nodes and return a list of the nodes
    containing the desired data tag."""

    response_body = {"message": None}
    status_code = None

    try:
        body = json.loads(request.data)
        grid_nodes = network_manager.connected_nodes()

        # Perform requests (HTTP) to all known nodes looking for the desired data tag
        match_grid_nodes = []
        for node in grid_nodes:
            try:
                response = requests.post(
                    grid_nodes[node] + "/data-centric/search",
                    data=json.dumps({"query": body["query"]}),
                ).content
            except requests.exceptions.ConnectionError:
                continue
            response = json.loads(response)

            if response["content"]:
                match_grid_nodes.append((node, grid_nodes[node]))

        # Return a list[(id, address)]  with all grid nodes that have the desired data
        response_body = match_grid_nodes
        status_code = 200

    except (ValueError, KeyError) as e:
        response_body["message"] = INVALID_JSON_FORMAT_MESSAGE
        status_code = 400
    except Exception as e:
        response_body["message"] = str(e)
        status_code = 500  # Internal Server Error

    print("Response body: ", response_body)
    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


def _get_model_hosting_nodes(model_id):
    """Search all nodes if they are currently hosting the model.

    Args:
        model_id: ID of the model.

    Returns:
        list: An array of the nodes currently hosting the model.
    """
    grid_nodes = network_manager.connected_nodes()
    match_nodes = []
    for node in grid_nodes:
        try:
            response = requests.get(grid_nodes[node] + "/data-centric/models/").content
        except requests.exceptions.ConnectionError:
            continue
        response = json.loads(response)
        if model_id in response.get("models", []):
            match_nodes.append((node, grid_nodes[node]))

    return match_nodes
