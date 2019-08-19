"""
    All Gateway routes (REST API).
"""

from flask import render_template, Response, request
from . import main
import json

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
        response = requests.post(
            grid_nodes[node] + "/search", data=json.dumps({"query": body["query"]})
        ).content
        response = json.loads(response)
        # If contains
        if response["content"]:
            match_grid_nodes.append((node, grid_nodes[node]))

    # Return a list[ (id, address) ]  with all grid nodes that have the desired data
    return Response(
        json.dumps(match_grid_nodes), status=200, mimetype="application/json"
    )
