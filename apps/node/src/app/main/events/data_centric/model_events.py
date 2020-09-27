# Standard Python imports
import json

import syft as sy

# External imports
import torch as th
from flask_login import current_user
from syft.codes import RESPONSE_MSG
from syft.generic.pointers.pointer_tensor import PointerTensor

# Local imports
from ... import hook, local_worker
from ...core.codes import MSG_FIELD
from ...data_centric.auth import authenticated_only
from ...data_centric.persistence import model_controller
from ...data_centric.persistence.object_storage import recover_objects


@authenticated_only
def host_model(message: dict) -> dict:
    """Save/Store a model into database.

    Args:
        message (dict) : Dict containing a serialized model and model's metadata.
    Response:
        response (dict) : Node's response.
    """
    encoding = message["encoding"]
    model_id = message[MSG_FIELD.MODEL_ID]
    allow_download = message[MSG_FIELD.ALLOW_DOWNLOAD] == "True"
    allow_remote_inference = message[MSG_FIELD.ALLOW_REMOTE_INFERENCE] == "True"
    mpc = message[MSG_FIELD.MPC] == "True"
    serialized_model = message[MSG_FIELD.MODEL]

    # Encode the model accordingly
    serialized_model = serialized_model.encode(encoding)

    # save the model for later usage
    response = model_controller.save(
        current_user.worker,
        serialized_model,
        model_id,
        allow_download,
        allow_remote_inference,
        mpc,
    )
    return response


@authenticated_only
def delete_model(message: dict) -> dict:
    """Delete a model previously stored at database.

    Args:
        message (dict) : Model's id.
    Returns:
        response (dict) : Node's response.
    """
    model_id = message[MSG_FIELD.MODEL_ID]
    result = model_controller.delete(current_user.worker, model_id)
    return result


@authenticated_only
def get_models(message: dict) -> dict:
    """Get a list of stored models.

    Returns:
        response (dict) : List of models stored at this node.
    """
    model_list = model_controller.models(current_user.worker)
    return model_list


@authenticated_only
def run_inference(message: dict) -> dict:
    """Run dataset inference with a specifc model stored in this node.

    Args:
        message (dict) : Serialized dataset, model id and dataset's metadata.
    Returns:
        response (dict) : Model's inference.
    """
    ## If worker is empty, load previous database tensors.
    if not current_user.worker._objects:
        recover_objects(current_user.worker)

    response = model_controller.get(current_user.worker, message[MSG_FIELD.MODEL_ID])

    if response[RESPONSE_MSG.SUCCESS]:

        # If model exists but not allow remote inferences
        if not response[MSG_FIELD.PROPERTIES][MSG_FIELD.ALLOW_REMOTE_INFERENCE]:
            return {
                MSG_FIELD.SUCCESS: False,
                "not_allowed": True,
                RESPONSE_MSG.ERROR: "You're not allowed to run inferences on this model.",
            }

        model = response[MSG_FIELD.PROPERTIES][MSG_FIELD.MODEL]

        # serializing the data from GET request
        encoding = message["encoding"]
        serialized_data = message["data"].encode(encoding)
        data = sy.serde.deserialize(serialized_data)

        # If we're using a Plan we need to register the object
        # to the local worker in order to execute it
        current_user.worker._objects[data.id] = data

        # Some models returns tuples (GPT-2 / BERT / ...)
        # To avoid errors on detach method, we check the type of inference's result
        model_output = model(data)

        # It the model is a plan, it'll receive a tensor wrapper as a model_output.
        while model_output.is_wrapper or isinstance(model_output, PointerTensor):
            model_output = model_output.get()

        if isinstance(model_output, tuple):
            predictions = model_output[0].detach().numpy().tolist()
        else:
            predictions = model_output.detach().numpy().tolist()

        # We can now remove data from the objects
        del data
        return {RESPONSE_MSG.SUCCESS: True, RESPONSE_MSG.INFERENCE_RESULT: predictions}
    else:
        return response
