"""This file exists to provide one common place for all grid node http
requests."""

import json
import os
import sys

import syft as sy
from flask import Response, render_template, request, send_from_directory
from flask_cors import cross_origin
from syft.codes import RESPONSE_MSG
from syft.grid.clients.data_centric_fl_client import DataCentricFLClient

from ... import data_centric_routes, local_worker
from ...core.codes import MSG_FIELD
from ...data_centric.persistence import model_controller

# ======= WEB ROUTES ======


@data_centric_routes.route("/detailed-models-list/")
def list_models_with_details():
    """Generates a detailed list of models currently saved at the worker.

    Returns:
        Response : List of models (and their properties) stored at this node.
    """
    model_ids = model_controller.models(local_worker)["models"]
    models = list()
    for id in model_ids:
        model = model_controller.get(local_worker, id)
        model_data = model_controller.get(local_worker, id)[MSG_FIELD.PROPERTIES]
        models.append(
            {
                "id": id,
                MSG_FIELD.SIZE: "{}KB".format(
                    sys.getsizeof(model_data[MSG_FIELD.MODEL]) / 1000
                ),
                MSG_FIELD.ALLOW_DOWNLOAD: model_data[MSG_FIELD.ALLOW_DOWNLOAD],
                MSG_FIELD.ALLOW_REMOTE_INFERENCE: model_data[
                    MSG_FIELD.ALLOW_REMOTE_INFERENCE
                ],
                MSG_FIELD.MPC: model_data[MSG_FIELD.MPC],
            }
        )
    return Response(
        json.dumps({RESPONSE_MSG.SUCCESS: True, RESPONSE_MSG.MODELS: models}),
        status=200,
        mimetype="application/json",
    )


@data_centric_routes.route("/identity/")
def identity():
    """Generates a response with the name of this node.

    Returns:
        Response : Name of node
    """

    return Response(
        json.dumps({RESPONSE_MSG.SUCCESS: True, "identity": local_worker.id}),
        status=200,
        mimetype="application/json",
    )


@data_centric_routes.route("/status/")
def show_status():
    """Generates a response with the status of this node. if the nodes is
    connected to workers, the status is online.

    Returns:
        Response : Status of node
    """

    connected_workers = filter(
        lambda x: isinstance(x, DataCentricFLClient),
        local_worker._known_workers.values(),
    )
    ids = map(lambda x: x.id, connected_workers)

    status = "OpenGrid"

    return Response(
        json.dumps({RESPONSE_MSG.SUCCESS: True, "status": status}),
        status=200,
        mimetype="application/json",
    )


@data_centric_routes.route("/workers/")
def list_workers():
    """Generates a list of remote nodes directly connected to this node.

    Returns:
        Response : List of node's ids.
    """
    connected_workers = filter(
        lambda x: isinstance(x, DataCentricFLClient),
        local_worker._known_workers.values(),
    )
    ids = map(lambda x: x.id, connected_workers)
    response_body = {RESPONSE_MSG.SUCCESS: True, "workers": list(ids)}
    return Response(json.dumps(response_body), status=200, mimetype="application/json")


# ======= WEB ROUTES END ======

# ======= REST API =======


@data_centric_routes.route("/models/", methods=["GET"])
@cross_origin()
def list_models():
    """Generates a list of models currently saved at the worker.

    Returns:
         Response : List of model's ids stored at this node.
    """
    return Response(
        json.dumps(model_controller.models(local_worker)),
        status=200,
        mimetype="application/json",
    )


@data_centric_routes.route("/serve-model/", methods=["POST"])
@cross_origin()
def serve_model():
    """Host an AI model Uploading and registering it by an specific ID.

    Returns:
        Response : A json structure informing if hosting was succeed.
    """
    encoding = request.form["encoding"]
    model_id = request.form[MSG_FIELD.MODEL_ID]
    allow_download = request.form[MSG_FIELD.ALLOW_DOWNLOAD] == "True"
    allow_remote_inference = request.form[MSG_FIELD.ALLOW_REMOTE_INFERENCE] == "True"
    mpc = request.form[MSG_FIELD.MPC] == "True"

    if request.files:
        # If model is large, receive it by a stream channel
        try:
            serialized_model = request.files[MSG_FIELD.MODEL].read().decode("utf-8")
        except UnicodeDecodeError:
            serialized_model = request.files[MSG_FIELD.MODEL].read().decode("latin-1")
    else:
        # If model is small, receive it by a standard json
        serialized_model = request.form[MSG_FIELD.MODEL]

    # Encode the model accordingly
    serialized_model = serialized_model.encode(encoding)

    # save the model for later usage
    response = model_controller.save(
        local_worker,
        serialized_model,
        model_id,
        allow_download,
        allow_remote_inference,
        mpc,
    )

    if response[RESPONSE_MSG.SUCCESS]:
        return Response(json.dumps(response), status=200, mimetype="application/json")
    else:
        return Response(json.dumps(response), status=409, mimetype="application/json")


@data_centric_routes.route("/dataset-tags", methods=["GET"])
@cross_origin()
def get_available_tags():
    """Returns all tags stored in this node. Can be very useful to know what
    datasets this node contains.

    Returns:
        Response : List of dataset tags stored at this node.
    """
    available_tags = set()
    objs = local_worker._objects

    for obj in objs.values():
        if obj.tags:
            available_tags.update(set(obj.tags))

    return Response(
        json.dumps(list(available_tags)), status=200, mimetype="application/json"
    )


@data_centric_routes.route("/search-encrypted-models", methods=["POST"])
@cross_origin()
def search_encrypted_models():
    """Check if exist some encrypted model hosted on this node using a specific
    model_id, if found, return JSON with a list of workers/crypto_provider."""
    try:
        body = json.loads(request.data)
    except json.decoder.JSONDecodeError:
        return Response(
            json.dumps({"error": "Invalid payload format"}),
            status=400,
            mimetype="application/json",
        )

    # Check json fields
    if body.get(MSG_FIELD.MODEL_ID):
        # Search model_id on node objects
        model = local_worker._objects.get(body.get(MSG_FIELD.MODEL_ID))

        # If found model is a plan
        if isinstance(model, sy.Plan):

            workers = set()
            # Check every state used by this plan
            for state_id in model.state.state_placeholders:
                obj = local_worker._objects.get(state_id.id.value)
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
            response = {RESPONSE_MSG.ERROR: "Model ID not found!"}
            response_status = 404
    # JSON without model_id field
    else:
        response = {RESPONSE_MSG.ERROR: "Invalid payload format"}
        response_status = 400
    return Response(
        json.dumps(response), status=response_status, mimetype="application/json"
    )


@data_centric_routes.route("/search", methods=["POST"])
@cross_origin()
def search_dataset_tags():
    """Search for a specific dataset tag stored at this node."""
    body = json.loads(request.data)

    # Invalid body
    if "query" not in body:
        return Response("", status=400, mimetype="application/json")

    # Search for desired datasets that belong to this node
    results = local_worker.search(body["query"])

    body_response = {"content": False}
    if len(results):
        body_response["content"] = True

    return Response(json.dumps(body_response), status=200, mimetype="application/json")


# ======= REST API END =======
