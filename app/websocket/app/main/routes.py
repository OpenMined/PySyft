"""
This file exists to provide one common place for all grid node http requests.
"""
import binascii
import json
import sys
import os

from flask import render_template
from flask import Response
from flask import request, send_from_directory

import syft as sy
from requests_toolbelt import MultipartEncoder
from flask_cors import cross_origin

from . import html, local_worker
from . import model_manager as mm
from .local_worker_utils import register_obj, get_objs


# Suport for sending big models over the wire back to a
# worker
MODEL_LIMIT_SIZE = (1024 ** 2) * 100  # 100MB

# ======= WEB ROUTES ======


@html.route("/favicon.ico")
def favicon():
    return send_from_directory(
        os.path.join(html.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon",
    )


@html.route("/", methods=["GET"])
def index():
    """Index page."""
    return render_template("index.html")


@html.route("/detailed_models_list/")
def list_models_with_details():
    """Generates a detailed list of models currently saved at the worker"""
    return Response(
        json.dumps(mm.list_models(detailed_list=True)),
        status=200,
        mimetype="application/json",
    )


@html.route("/workers/")
def list_workers():
    return Response(
        json.dumps(mm.list_workers()), status=200, mimetype="application/json"
    )


# ======= WEB ROUTES END ======

# ======= REST API =======


@html.route("/models/", methods=["GET"])
@cross_origin()
def list_models():
    """Generates a list of models currently saved at the worker"""
    return Response(
        json.dumps(mm.list_models()), status=200, mimetype="application/json"
    )


@html.route("/is_model_copy_allowed/<model_id>", methods=["GET"])
@cross_origin()
def is_model_copy_allowed(model_id):
    """Generates a list of models currently saved at the worker"""
    return Response(
        json.dumps(mm.is_model_copy_allowed(model_id)),
        status=200,
        mimetype="application/json",
    )


## Will be removed when sockets can download huge models
@html.route("/get_model/<model_id>", methods=["GET"])
@cross_origin()
def get_model(model_id):
    """ Try to download a specific model if allowed. """

    # If not Allowed
    check = mm.is_model_copy_allowed(model_id)
    response = {}
    if not check["success"]:  # If not allowed
        if check["error"] == mm.MODEL_NOT_FOUND_MSG:
            status_code = 404  # Not Found
            response["error"] = mm.Model_NOT_FOUND_MSG
        else:
            status_code = 403  # Forbidden
            response["error"] = mm.NOT_ALLOWED_TO_DOWNLOAD_MSG
        return Response(
            json.dumps(response), status=status_code, mimetype="application/json"
        )

    # If allowed
    result = mm.get_serialized_model_with_id(model_id)

    if result["success"]:
        # Use correct encoding
        response = {"serialized_model": result["serialized_model"].decode("ISO-8859-1")}

        # If model is large split it in multiple parts
        if sys.getsizeof(response["serialized_model"]) >= MODEL_LIMIT_SIZE:
            form = MultipartEncoder(response)
            return Response(form.to_string(), mimetype=form.content_type)
        else:
            return Response(
                json.dumps(response), status=200, mimetype="application/json"
            )


## Will be removed when sockets can upload huge models
@html.route("/serve-model/", methods=["POST"])
@cross_origin()
def serve_model():
    encoding = request.form["encoding"]
    model_id = request.form["model_id"]
    allow_download = request.form["allow_download"] == "True"
    allow_remote_inference = request.form["allow_remote_inference"] == "True"

    if request.files:
        # If model is large, receive it by a stream channel
        serialized_model = request.files["model"].read().decode("utf-8")
    else:
        # If model is small, receive it by a standard json
        serialized_model = request.form["model"]

    # Encode the model accordingly
    serialized_model = serialized_model.encode(encoding)

    # save the model for later usage
    response = mm.save_model(
        serialized_model, model_id, allow_download, allow_remote_inference
    )
    if response["success"]:
        return Response(json.dumps(response), status=200, mimetype="application/json")
    else:
        return Response(json.dumps(response), status=409, mimetype="application/json")


@html.route("/dataset-tags", methods=["GET"])
@cross_origin()
def get_available_tags():
    """ Returns all tags stored in this node. Can be very useful to know what datasets this node contains. """
    available_tags = set()
    objs = get_objs()

    for obj in objs.values():
        if obj.tags:
            available_tags.update(set(obj.tags))

    return Response(
        json.dumps(list(available_tags)), status=200, mimetype="application/json"
    )


@html.route("/search-encrypted-models", methods=["POST"])
@cross_origin()
def search_encrypted_models():
    """ Check if exist some encrypted model hosted on this node using a specific model_id, if found,
        return JSON with a list of workers/crypto_provider.
    """
    try:
        body = json.loads(request.data)
    except json.decoder.JSONDecodeError:
        return Response(
            json.dumps({"error": "Invalid payload format"}),
            status=400,
            mimetype="application/json",
        )

    # Check json fields
    if body.get("model_id"):
        # Search model_id on node objects
        model = local_worker._objects.get(body.get("model_id"))

        # If found model is a plan
        if isinstance(model, sy.Plan):

            workers = set()
            # Check every state used by this plan
            for state_id in model.state_ids:
                obj = local_worker._objects.get(state_id)
                # Decrease in Tensor Hierarchy (we want be a AdditiveSharingTensor to recover workers/crypto_provider addresses)
                while not isinstance(obj, sy.AdditiveSharingTensor):
                    obj = obj.child

                # Get a list of tuples (worker_id, worker_address)
                worker_urls = map(
                    lambda x: (x, local_worker._known_workers.get(x).address),
                    obj.child.keys(),
                )
                workers.update(set(worker_urls))

                # Get crypto_provider id/address
                if obj.crypto_provider:
                    crypto_provider = [
                        obj.crypto_provider.id,
                        local_worker._known_workers.get(obj.crypto_provider.id).address,
                    ]

            response = {
                "workers": list(workers),
                "crypto_provider": list(crypto_provider),
            }
            response_status = 200
        else:
            response = {"error": "Model ID not found!"}
            response_status = 404
    # JSON without model_id field
    else:
        response = {"error": "Invalid payload format"}
        response_status = 400
    return Response(
        json.dumps(response), status=response_status, mimetype="application/json"
    )


@html.route("/search", methods=["POST"])
@cross_origin()
def search_dataset_tags():
    body = json.loads(request.data)

    # Invalid body
    if "query" not in body:
        return Response("", status=400, mimetype="application/json")

    # Search for desired datasets that belong to this node
    results = local_worker.search(*body["query"])

    body_response = {"content": False}
    if len(results):
        body_response["content"] = True

    return Response(json.dumps(body_response), status=200, mimetype="application/json")


# ======= REST API END =======
