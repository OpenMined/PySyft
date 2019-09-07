"""
This file exists to provide one common place for all grid node http requests.
"""
import binascii
import json

from flask import render_template
from flask import Response
from flask import request
import syft as sy

from . import main
from . import hook


models = {}


@main.route("/identity/")
def is_this_an_opengrid_node():
    """This exists because in the automation scripts which deploy nodes,
    there's an edge case where the 'node already exists' but sometimes it
    can be an app that does something totally different. So we want to have
    some endpoint which just casually identifies this server as an OpenGrid
    server."""
    return "OpenGrid"


@main.route("/models/", methods=["GET"])
def list_models():
    return Response(
        json.dumps({"models": list(models.keys())}),
        status=202,
        mimetype="application/json",
    )


@main.route("/models/<model_id>", methods=["GET"])
def model_inference(model_id):
    if model_id not in models:
        return Response(
            json.dumps({"UnknownModel": "Unknown model {}".format(model_id)}),
            status=404,
            mimetype="application/json",
        )

    model = models[model_id]
    encoding = request.form["encoding"]
    serialized_data = request.form["data"].encode(encoding)
    data = sy.serde.deserialize(serialized_data)

    # If we're using a Plan we need to register the object
    # to the local worker in order to execute it
    sy.hook.local_worker.register_obj(data)

    response = model(data).detach().numpy().tolist()

    # We can now remove data from the objects
    del data

    return Response(
        json.dumps({"prediction": response}), status=200, mimetype="application/json"
    )


@main.route("/serve-model/", methods=["POST"])
def serve_model():
    encoding = request.form["encoding"]
    serialized_model = request.form["model"].encode(encoding)
    model_id = request.form["model_id"]

    deserialized_model = sy.serde.deserialize(serialized_model)

    # TODO store this in a local database
    if model_id in models:
        return Response(
            json.dumps(
                {
                    "error": "Model ID should be unique. There is already a model being hosted with this id."
                }
            ),
            status=404,
            mimetype="application/json",
        )

    models[model_id] = deserialized_model

    return Response(
        json.dumps({"success": True}), status=200, mimetype="application/json"
    )


@main.route("/", methods=["GET"])
def index():
    """Index page."""
    return render_template("index.html")


@main.route("/dataset-tags", methods=["GET"])
def get_available_tags():
    """ Returns all tags stored in this node. Can be very useful to know what datasets this node contains. """
    available_tags = set()
    objs = hook.local_worker._objects

    for key, obj in objs.items():
        if obj.tags:
            available_tags.update(set(obj.tags))

    return Response(
        json.dumps(list(available_tags)), status=200, mimetype="application/json"
    )


@main.route("/search", methods=["POST"])
def search_dataset_tags():
    body = json.loads(request.data)

    # Invalid body
    if "query" not in body:
        return Response("", status=400, mimetype="application/json")

    # Search for desired datasets that belong to this node
    results = hook.local_worker.search(*body["query"])

    body_response = {"content": False}
    if len(results):
        body_response["content"] = True

    return Response(json.dumps(body_response), status=200, mimetype="application/json")
