import json
import torch as th

import syft as sy
from syft.codes import RESPONSE_MSG
from syft.generic.pointers.pointer_tensor import PointerTensor
from flask_login import current_user


from .. import local_worker, hook
from ..persistence import model_controller
from ..persistence.object_storage import recover_objects
from ..codes import MODEL
from ..auth import authenticated_only


@authenticated_only
def host_model(message: dict) -> str:
    """ Save/Store a model into database.

        Args:
            message (dict) : Dict containing a serialized model and model's metadata.
        Response:
            response (str) : Node's response.
    """
    encoding = message["encoding"]
    model_id = message[MODEL.ID]
    allow_download = message[MODEL.ALLOW_DOWNLOAD] == "True"
    allow_remote_inference = message[MODEL.ALLOW_REMOTE_INFERENCE] == "True"
    mpc = message[MODEL.MPC] == "True"
    serialized_model = message[MODEL.MODEL]

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
    return json.dumps(response)


@authenticated_only
def delete_model(message: dict) -> str:
    """ Delete a model previously stored at database.

        Args:
            message (dict) : Model's id.
        Returns:
            response (str) : Node's response.
    """
    model_id = message[MODEL.ID]
    result = model_controller.delete(current_user.worker, model_id)
    return json.dumps(result)


@authenticated_only
def get_models(message: dict) -> str:
    """ Get a list of stored models.

        Returns:
            response (str) : List of models stored at this node.
    """
    model_list = model_controller.models(current_user.worker)
    return json.dumps(model_list)


@authenticated_only
def run_inference(message: dict) -> str:
    """ Run dataset inference with a specifc model stored in this node.

        Args:
            message (dict) : Serialized dataset, model id and dataset's metadata.
        Returns:
            response (str) : Model's inference.
    """
    ## If worker is empty, load previous database tensors.
    if not current_user.worker._objects:
        recover_objects(current_user.worker)

    response = model_controller.get(current_user.worker, message[MODEL.ID])

    if response[RESPONSE_MSG.SUCCESS]:

        # If model exists but not allow remote inferences
        if not response[MODEL.PROPERTIES][MODEL.ALLOW_REMOTE_INFERENCE]:
            return json.dumps(
                {
                    RESPONSE_MSG.SUCCESS: False,
                    "not_allowed": True,
                    RESPONSE_MSG.ERROR: "You're not allowed to run inferences on this model.",
                }
            )

        model = response[MODEL.PROPERTIES][MODEL.MODEL]

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
        return json.dumps(
            {RESPONSE_MSG.SUCCESS: True, RESPONSE_MSG.INFERENCE_RESULT: predictions}
        )
    else:
        return json.dumps(response)
