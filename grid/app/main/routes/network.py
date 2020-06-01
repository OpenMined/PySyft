# PyGrid imports
from .. import main
from ..models import model_manager
from ..network import network_manager
from ..processes import process_manager
from ..codes import RESPONSE_MSG


# General imports
import io
import os
import json
import random
import requests
from flask import render_template, Response, request, current_app, send_file


# All grid nodes registered at grid network will be stored here
grid_nodes = {}
SMPC_HOST_CHUNK = 4  # Minimum nodes required to host an encrypted model
INVALID_JSON_FORMAT_MESSAGE = (
    "Invalid JSON format."  # Default message used to report Invalid JSON format.
)


@main.route("/join", methods=["POST"])
def join_grid_node():
    """ Register a new grid node at grid network.
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


@main.route("/connected-nodes", methods=["GET"])
def get_connected_nodes():
    """ Get a list of connected nodes. """
    grid_nodes = network_manager.connected_nodes()
    return Response(
        json.dumps({"grid-nodes": list(grid_nodes.keys())}),
        status=200,
        mimetype="application/json",
    )


@main.route("/delete-node", methods=["DELETE"])
def delete_grid_node():
    """ Delete a grid node at grid network"""

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


@main.route("/choose-encrypted-model-host", methods=["GET"])
def choose_encrypted_model_host():
    """ Used to choose grid nodes to host an encrypted model
        PS: currently we perform this randomly
    """
    grid_nodes = network_manager.connected_nodes()
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
    grid_nodes = network_manager.connected_nodes()
    n_replica = current_app.config["N_REPLICA"]
    if not n_replica:
        n_replica = 1

    model_id = request.args.get("model_id")
    hosts_info = None

    # lookup the nodes already hosting this model to prevent hosting different model versions
    if model_id:
        hosts_info = _get_model_hosting_nodes(model_id)

    # no model id given or no hosting nodes found: randomly choose node
    if not hosts_info:
        hosts = random.sample(list(grid_nodes.keys()), n_replica)
        hosts_info = [(host, grid_nodes[host]) for host in hosts]

    return Response(json.dumps(hosts_info), status=200, mimetype="application/json")


@main.route("/search-encrypted-model", methods=["POST"])
def search_encrypted_model():
    """ Search for an encrypted plan model on the grid network, if found,
        returns host id, host address and SMPC workers infos.
    """

    response_body = {"message": None}
    status_code = None

    try:
        grid_nodes = network_manager.connected_nodes()
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


@main.route("/search-model", methods=["POST"])
def search_model():
    """ Search for a plain text model on the grid network. """

    response_body = {"message": None}
    status_code = None

    try:
        body = json.loads(request.data)

        model_id = body["model_id"]
        match_nodes = _get_model_hosting_nodes(model_id)

        # It returns a list[ (id, address) ]  with all grid nodes that have the desired model
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


@main.route("/search-available-models", methods=["GET"])
def available_models():
    """ Get all available models on the grid network. Can be useful to know what models our grid network have. """
    grid_nodes = network_manager.connected_nodes()
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
    grid_nodes = network_manager.connected_nodes()
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
                    grid_nodes[node] + "/search",
                    data=json.dumps({"query": body["query"]}),
                ).content
            except requests.exceptions.ConnectionError:
                continue
            response = json.loads(response)
            # If contains
            if response["content"]:
                match_grid_nodes.append((node, grid_nodes[node]))

        # It returns a list[ (id, address) ]  with all grid nodes that have the desired data
        response_body = match_grid_nodes
        status_code = 200

    except (ValueError, KeyError) as e:
        response_body["message"] = INVALID_JSON_FORMAT_MESSAGE
        status_code = 400
    except Exception as e:
        response_body["message"] = str(e)
        status_code = 500  # Internal Server Error

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


@main.route("/get-model", methods=["GET"])
def get_model():
    """Request a download of a model"""

    response_body = {}
    status_code = None
    try:
        name = request.args.get("name", None)
        version = request.args.get("version", None)
        checkpoint = request.args.get("checkpoint", None)

        _fl_process = process_manager.get(name=name, version=version)
        _model = model_manager.get(fl_process_id=_fl_process.id)
        _model_checkpoint = model_manager.load(model_id=_model.id, id=checkpoint)

        return send_file(
            io.BytesIO(_model_checkpoint.values), mimetype="application/octet-stream"
        )

    except Exception as e:
        status_code = 500  # Internal Server Error
        response_body[RESPONSE_MSG.ERROR] = str(e)

    return Response(
        json.dumps(response_body), status=status_code, mimetype="application/json"
    )


def _get_model_hosting_nodes(model_id):
    """ Search all nodes if they are currently hosting the model.

    :param model_id: The model to search for
    :return: An array of the nodes currently hosting the model
    """
    grid_nodes = network_manager.connected_nodes()
    match_nodes = []
    for node in grid_nodes:
        try:
            response = requests.get(grid_nodes[node] + "/models/").content
        except requests.exceptions.ConnectionError:
            continue
        response = json.loads(response)
        if model_id in response.get("models", []):
            match_nodes.append((node, grid_nodes[node]))

    return match_nodes
