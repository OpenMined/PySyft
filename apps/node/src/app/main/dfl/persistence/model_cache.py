# Standard Python imports
import collections
from typing import List, Union

# External imports
from syft.serde import deserialize
from torch import nn as nn

# Local imports
from ...core.codes import MSG_FIELD


class ModelCache:
    """Handle models in memory."""

    def __init__(self):
        self.cache = dict()

    @property
    def models(self) -> List:
        """Returns a list of model ids stored at this cache."""
        return list(self.cache.keys())

    def contains(self, model_id: str) -> bool:
        """Checks if the given model_id is present in cache.

        Args:
            model_id (str): Unique id representing the model.
        Returns:
            True is present, else False.
        """
        return model_id in self.cache

    def get(self, model_id: str) -> Union[nn.Module, None]:
        """Checks the cache for a model.

        If model not found, returns None.
        Args:
            model_id (str): Unique id representing the model.
        Returns:
            An encoded model, else returns None.
        """
        if self.contains(model_id):
            return self.cache.get(model_id)

    def save(
        self,
        model,
        model_id: str,
        allow_download: bool,
        allow_remote_inference: bool,
        mpc: bool,
        serialized: bool = True,
    ) -> bool:
        """Saves the model to cache.

        Nothing happens if a model with the same id already exists.
        Args:
            model: The model object to be saved.
            model_id: The unique identifier associated with the model.
            serialized: If the model is serialized or not. If it is this method
            deserializes it.
        Returns:
            result : Boolean result  denoting if the given model was saved.
        """
        if not self.contains(model_id):
            if serialized:
                model = deserialize(model)

            # Store model
            self.cache[model_id] = {
                MSG_FIELD.MODEL: model,
                MSG_FIELD.ALLOW_DOWNLOAD: allow_download,
                MSG_FIELD.ALLOW_REMOTE_INFERENCE: allow_remote_inference,
                MSG_FIELD.MPC: mpc,
            }
            return True
        else:
            return False

    def remove(self, model_id: str) -> bool:
        """Deletes the given model_id from cache.

        Args:
            model_id : Unique id representing the model.
        Returns:
            result : Boolean result  denoting if the given model was deleted.
        """
        if self.contains(model_id):
            del self.cache[model_id]
            return True
        else:
            return False

    def clear(self) -> None:
        """Clears the cache."""
        self.cache = dict()
