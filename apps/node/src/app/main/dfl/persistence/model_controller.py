# Standard Python imports
from typing import Dict

# External imports
from syft import Plan
from syft.codes import RESPONSE_MSG
from syft.serde import deserialize

from ...core.codes import MSG_FIELD

# Local imports
from .model_storage import ModelStorage


class ModelController:
    """Controller Design Pattern to manage models on database/cache."""

    # Error Messages
    ID_CONFLICT_MSG = "Model id already exists."
    MODEL_NOT_FOUND_MSG = "Model not found."
    MODEL_DELETED_MSG = "Model deleted with success!"

    def __init__(self):
        self.model_storages = dict()

    def save(
        self,
        worker,
        serialized_model: bytes,
        model_id: str,
        allow_download: bool,
        allow_remote_inference: bool,
        mpc: bool = False,
    ) -> Dict:
        """Map and Save the desired model at database/cache.

        If persistent mode isn't enable, this model will be saved in cache memory.
        Args:
            worker: Worker that owns this model.
            serialized_model: Model serialized.
            model_id: Model's ID.
            allow_download: Flag to enable/disable download.
            allow_remote_inference: Flag to enable/disable remote inference.
            mpc: Flag used to identify if it is an encrypted model.
        Returns:
            response_msg: Dict response message.
        """
        storage = self.get_storage(worker)

        if storage.contains(model_id):
            return {
                RESPONSE_MSG.SUCCESS: False,
                RESPONSE_MSG.ERROR: ModelController.ID_CONFLICT_MSG,
            }

        # Saves a copy in the database
        storage.save_model(
            serialized_model, model_id, allow_download, allow_remote_inference, mpc
        )
        return {
            RESPONSE_MSG.SUCCESS: True,
            "message": "Model saved with id: " + model_id,
        }

    def get(self, worker, model_id: str) -> Dict:
        """Map and retrieves model from cache/database by model_id.

        If persistent mode isn't enable, this model will be searched only in the cache memory.
        Args:
            worker: Worker that owns this model.
            model_id: Model's ID.
        Returns:
            response_msg: Dict response message.
        """
        storage = self.get_storage(worker)
        if storage.contains(model_id):
            return {
                RESPONSE_MSG.SUCCESS: True,
                MSG_FIELD.PROPERTIES: storage.get(model_id),
            }
        else:
            return {
                RESPONSE_MSG.SUCCESS: False,
                RESPONSE_MSG.ERROR: ModelController.MODEL_NOT_FOUND_MSG,
            }

    def delete(self, worker, model_id: str) -> Dict:
        """Delete the specific model from cache/database.

        Args:
            worker: Worker that owns this model.
            model_id: Model's ID
        Returns:
            response_msg: Dict response message.
        """
        storage = self.get_storage(worker)

        # Build response
        response = {}
        response[RESPONSE_MSG.SUCCESS] = bool(storage.remove(model_id))

        # set log messages
        if response[RESPONSE_MSG.SUCCESS]:
            response["message"] = "Model deleted with success!"
        else:
            response[RESPONSE_MSG.ERROR] = "Model id not found on database!"

        return response

    def models(self, worker) -> Dict:
        """Retrieves a list of model ids hosted by this worker.

        If persistent mode isn't enable,
        this method won't return model id's stored in the database.
        Args:
            worker: Worker that owns this model.
        Returns:
            response_msg: Dict response message.
        """
        storage = self.get_storage(worker)
        return {RESPONSE_MSG.SUCCESS: True, RESPONSE_MSG.MODELS: storage.models}

    def get_storage(self, worker) -> ModelStorage:
        """Returns the storage instance of an specific worker.

        Args:
            worker: Owner of this storage.
        Returns:
            storage: Worker's model Storage instance.
        """
        if worker.id in self.model_storages:
            storage = self.model_storages[worker.id]
        else:
            storage = self._new_storage(worker)
        return storage

    def _new_storage(self, worker) -> ModelStorage:
        """Create a new storage instance.

        Args:
            worker: Worker that will own this instance.
        Returns:
            new_storage: New storage instance.
        """
        new_storage = ModelStorage(worker)
        self.model_storages[worker.id] = new_storage
        return new_storage
