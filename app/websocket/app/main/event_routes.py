import json
from . import local_worker, hook
import syft as sy
import torch as th
from . import model_manager as mm
from .local_worker_utils import register_obj, get_objs
from grid import WebsocketGridClient
import sys

# Suport for sending big models over the wire back to a
# worker
MODEL_LIMIT_SIZE = (1024 ** 2) * 64  # 64MB


def get_node_id(message):
    return json.dumps({"id": local_worker.id})


def connect_grid_nodes(message):
    if message["id"] not in local_worker._known_workers:
        worker = WebsocketGridClient(hook, address=message["address"], id=message["id"])
    return json.dumps({"status": "Succesfully connected."})


def socket_ping(message):
    return json.dumps({"alive": "True"})


def syft_command(message):
    response = local_worker._message_router[message["msg_type"]](message["content"])
    payload = sy.serde.serialize(response, force_no_serialization=True)
    return json.dumps({"type": "command-response", "response": payload})


def host_model(message):
    encoding = message["encoding"]
    model_id = message["model_id"]
    allow_download = message["allow_download"] == "True"
    allow_remote_inference = message["allow_remote_inference"] == "True"

    serialized_model = message["model"]

    # Encode the model accordingly
    serialized_model = serialized_model.encode(encoding)

    # save the model for later usage
    response = mm.save_model(
        serialized_model, model_id, allow_download, allow_remote_inference
    )
    return json.dumps(response)


def delete_model(message):
    model_id = message["model_id"]
    result = mm.delete_model(model_id)
    return json.dumps(result)


def get_models(message):
    return json.dumps(mm.list_models())


def download_model(message):
    model_id = message["model_id"]

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
        response["success"] = False
        return json.dumps(response)

    # If allowed
    result = mm.get_serialized_model_with_id(model_id)

    if result["success"]:
        # Use correct encoding
        response = {"serialized_model": result["serialized_model"].decode("ISO-8859-1")}
        if sys.getsizeof(response["serialized_model"]) >= MODEL_LIMIT_SIZE:
            # Forward to HTTP method
            # TODO: Implement download of huge models using sockets
            return json.dumps({"success": False})
        else:
            return json.dumps(response)


def run_inference(message):
    response = mm.get_model_with_id(message["model_id"])
    if response["success"]:
        model = response["model"]

        # serializing the data from GET request
        encoding = message["encoding"]
        serialized_data = message["data"].encode(encoding)
        data = sy.serde.deserialize(serialized_data)

        # If we're using a Plan we need to register the object
        # to the local worker in order to execute it
        register_obj(data)

        # Some models returns tuples (GPT-2 / BERT / ...)
        # To avoid errors on detach method, we check the type of inference's result
        model_output = model(data)
        if isinstance(model_output, tuple):
            predictions = model_output[0].detach().numpy().tolist()
        else:
            predictions = model_output.detach().numpy().tolist()

        # We can now remove data from the objects
        del data
        return json.dumps({"success": True, "prediction": predictions})
    else:
        return json.dumps(response)
