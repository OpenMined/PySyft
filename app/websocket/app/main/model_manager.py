import collections
import pickle
import os


from sqlalchemy.exc import SQLAlchemyError, IntegrityError
import syft as sy
import torch as th

from .persistence.models import db, TorchModel, TorchTensor
from .local_worker_utils import get_obj, register_obj


# ============= Global variables ========================

# Store models in memory
model_cache = dict()

# Local model abstraction
ModelTuple = collections.namedtuple(
    "ModelTuple", ["model_obj", "allow_download", "allow_remote_inference"]
)

# Error messages
MODEL_DELETED_MSG = "Model deleted with success!"
MODEL_NOT_FOUND_MSG = "Model not found."
NOT_ALLOWED_TO_RUN_INFERENCE_MSG = "You're not allowed to run inference on this model."
NOT_ALLOWED_TO_DOWNLOAD_MSG = "You're not allowed to download this model."

# ============== Cache related functions =================


def _clear_cache():
    """Clears the cache."""
    global model_cache
    model_cache = dict()


def _is_model_in_cache(model_id: str):
    """Checks if the given model_id is present in cache.

    Args:
        model_id (str): Unique id representing the model.

    Returns:
        True is present, else False.
    """
    return model_id in model_cache


def _get_model_from_cache(model_id: str):
    """Checks the cache for a model. If model not found, returns None.

    Args:
        model_id (str): Unique id representing the model.

    Returns:
        An encoded model, else returns None.
    """
    return model_cache.get(model_id)


def _save_model_to_cache(
    model,
    model_id: str,
    allow_download: bool,
    allow_remote_inference: bool,
    serialized: bool = True,
):
    """Saves the model to cache. Nothing happens if a model with the same id already exists.

    Args:
        model: The model object to be saved.
        model_id (str): The unique identifier associated with the model.
        serialized: If the model is serialized or not. If it is this method
            deserializes it.
    """
    if not _is_model_in_cache(model_id):
        if serialized:
            model = sy.serde.deserialize(model)
        model_cache[model_id] = ModelTuple(
            model_obj=model,
            allow_download=allow_download,
            allow_remote_inference=allow_remote_inference,
        )


def _remove_model_from_cache(model_id: str):
    """Deletes the given model_id from cache.

    Args:
        model_id (str): Unique id representing the model.
    """
    if _is_model_in_cache(model_id):
        del model_cache[model_id]


# ============== DB related functions =================


def _save_model_in_db(
    serialized_model: bytes,
    model_id: str,
    allow_download: bool,
    allow_remote_inference: bool,
):
    db.session.remove()
    db.session.add(
        TorchModel(
            id=model_id,
            model=serialized_model,
            allow_download=allow_download,
            allow_remote_inference=allow_remote_inference,
        )
    )
    db.session.commit()


def _save_states_in_db(model):
    tensors = []
    for state_id in model.state_ids:
        tensor = get_obj(state_id)
        tensors.append(TorchTensor(id=state_id, object=tensor.data))

    db.session.add_all(tensors)
    db.session.commit()


def _get_all_models_in_db():
    return db.session.query(TorchModel).all()


def _get_model_from_db(model_id: str):
    return db.session.query(TorchModel).get(model_id)


def _retrieve_state_from_db(model):
    for state_id in model.state_ids:
        result = db.session.query(TorchTensor).get(state_id)
        register_obj(result.object, state_id)


def _remove_model_from_db(model_id):
    result = _get_model_from_db(model_id)
    db.session.delete(result)
    db.session.commit()


# ================ Public functions ====================


def list_models():
    """Returns a dict of currently existing models. Will always fetch from db.

    Returns:
        A dict with structure: {"success": Bool, "models":[model list]}.
        On error returns dict: {"success": Bool, "error": error message}.
    """
    try:
        result = _get_all_models_in_db()
        model_ids = [model.id for model in result]
        return {"success": True, "models": model_ids}
    except SQLAlchemyError as e:
        return {"success": False, "error": str(e)}


def save_model(
    serialized_model: bytes,
    model_id: str,
    allow_download: bool,
    allow_remote_inference: bool,
):
    """Saves the model for later usage.

    Args:
        serialized_model (bytes): The model object to be saved.
        model_id (str): The unique identifier associated with the model.
        allow_download (bool): If the model can be copied by a worker.
        allow_remote_inference (bool): If a worker can run inference on the given model.

    Returns:
        A dict with structure: {"success": Bool, "message": "Model Saved: {model_id}"}.
        On error returns dict: {"success": Bool, "error": error message}.
    """
    if _is_model_in_cache(model_id):
        # Model already exists
        return {
            "success": False,
            "error": "Model with id: {} already eixsts.".format(model_id),
        }
    try:
        # Saves a copy in the database
        _save_model_in_db(
            serialized_model, model_id, allow_download, allow_remote_inference
        )

        # Also save a copy in cache
        model = sy.serde.deserialize(serialized_model)
        _save_model_to_cache(
            model, model_id, allow_download, allow_remote_inference, serialized=False
        )

        # If the model is a Plan we also need to store
        # the state tensors
        if isinstance(model, sy.Plan):
            _save_states_in_db(model)

        return {"success": True, "message": "Model saved with id: " + model_id}
    except (SQLAlchemyError, IntegrityError) as e:
        if type(e) is IntegrityError:
            # The model is already present within the db.
            # But missing from cache. Try to fetch the model and save to cache.
            return get_model_with_id(model_id)
        return {"success": False, "error": str(e)}


def get_model_with_id(model_id: str):
    """Returns a model with given model id.

    Args:
        model_id (str): The unique identifier associated with the model.

    Returns:
        A dict with structure: {"success": Bool, "model": serialized model object}.
        On error returns dict: {"success": Bool, "error": error message }.
    """
    if _is_model_in_cache(model_id):
        # Model already exists
        cache_model = _get_model_from_cache(model_id)
        if cache_model.allow_remote_inference:
            return {"success": True, "model": cache_model.model_obj}
        else:
            return {
                "success": False,
                "not_allowed": True,
                "error": NOT_ALLOWED_TO_RUN_INFERENCE_MSG,
            }
    try:
        result = _get_model_from_db(model_id)
        if result:
            model = sy.serde.deserialize(result.model)

            # If the model is a Plan we also need to retrieve
            # the state tensors
            if isinstance(model, sy.Plan):
                _retrieve_state_from_db(model)

            # Save model in cache
            _save_model_to_cache(
                model,
                model_id,
                result.allow_download,
                result.allow_remote_inference,
                serialized=False,
            )
            if result.allow_remote_inference:
                return {"success": True, "model": model}
            else:
                return {
                    "success": False,
                    "not_allowed": True,
                    "error": NOT_ALLOWED_TO_RUN_INFERENCE_MSG,
                }

        else:
            return {"success": False, "error": MODEL_NOT_FOUND_MSG}
    except SQLAlchemyError as e:
        return {"success": False, "error": str(e)}


def get_serialized_model_with_id(model_id: str):
    """Returns a serialized_model with given model id.

    Args:
        model_id (str): The unique identifier associated with the model.

    Returns:
        A dict with structure: {"success": Bool, "model": serialized model object}.
        On error returns dict: {"success": Bool, "error": error message }.
    """
    if _is_model_in_cache(model_id):
        # Model already exists
        cache_model = _get_model_from_cache(model_id)
        if cache_model.allow_download:
            return {
                "success": True,
                "serialized_model": sy.serde.serialize(cache_model.model_obj),
            }
        else:
            return {
                "success": False,
                "not_allowed": True,
                "error": NOT_ALLOWED_TO_DOWNLOAD_MSG,
            }
    try:
        result = _get_model_from_db(model_id)
        if result:
            model = sy.serde.deserialize(result.model)

            # If the model is a Plan we also need to retrieve
            # the state tensors
            if isinstance(model, sy.Plan):
                _retrieve_state_from_db(model)

            # Save model in cache
            _save_model_to_cache(
                model,
                model_id,
                result.allow_download,
                result.allow_remote_inference,
                serialized=False,
            )

            if result.allow_download:
                return {"success": True, "serialized_model": result.model}
            else:
                return {
                    "success": False,
                    "not_allowed": True,
                    "error": NOT_ALLOWED_TO_DOWNLOAD_MSG,
                }
        else:
            return {"success": False, "error": MODEL_NOT_FOUND_MSG}
    except SQLAlchemyError as e:
        return {"success": False, "error": str(e)}


def delete_model(model_id: str):
    """Deletes the given model id. If it is present.

    Args:
        model_id (str): The unique identifier associated with the model.

    Returns:
        A dict with structure: {"success": Bool, "message": "Model Deleted: {model_id}"}.
        On error returns dict: {"success": Bool, "error": {error message}}.
    """

    if not _get_model_from_db(model_id):
        return {"success": False, "error": MODEL_NOT_FOUND_MSG}

    try:
        # First del from cache
        _remove_model_from_cache(model_id)
        # Then del from db
        _remove_model_from_db(model_id)
        return {"success": True, "message": MODEL_DELETED_MSG}
    except SQLAlchemyError:
        # probably no model found in db.
        return {
            "success": False,
            "error": "Something went wrong while deleting the object, check if the object is listed at worker.models.",
        }


def is_model_copy_allowed(model_id: str):
    """Used to check a worker is allowed to run `download` in the model with this id."""
    result = _get_model_from_db(model_id)
    if result is None:
        return {"success": False, "error": MODEL_NOT_FOUND_MSG}
    elif result.allow_download:
        return {"success": True}
    else:
        return {"success": False, "error": NOT_ALLOWED_TO_DOWNLOAD_MSG}
