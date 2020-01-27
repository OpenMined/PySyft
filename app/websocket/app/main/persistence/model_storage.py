from typing import List, Dict, Union

from .database import db_instance
from .model_cache import ModelCache
from ..codes import MODEL
from syft.serde import serialize, deserialize
import hashlib


class ModelStorage:
    """ Manage all models hosted by an specific worker. """

    def __init__(self, worker):
        self.worker = worker
        self.cache = ModelCache()

    @property
    def id(self) -> str:
        """ Returns worker's id."""
        return self.worker.id

    @property
    def models(self) -> List:
        """ Returns a list of model ids hosted by this storage instance.
            If persistence mode isn't enabled, it will return models stored in cache memory.
            Returns:
                model_ids : List of model ids hosted by this storage instance.
        """
        # If persistence mode enabled
        if db_instance():
            key = self._generate_hash_key()
            size = db_instance().llen(key)
            model_ids = db_instance().lrange(key, 0, size)
            model_ids = [id.decode("utf-8") for id in model_ids]
            return model_ids

        return self.cache.models

    def save_model(
        self,
        serialized_model: bytes,
        model_id: str,
        allow_download: bool,
        allow_remote_inference: bool,
        mpc: bool,
    ):

        """ Save the desired model at database and load it in cache memory.
            Args:
                serialized_model: Model serialized.
                model_id: Model's ID.
                allow_download: Flag to enable/disable download.
                allow_remote_inference: Flag to enable/disable remote inference.
                mpc: Flag used to identify if it is an encrypted model.
        """
        # If persistence mode enabled
        if db_instance():
            key = self._generate_hash_key(model_id)
            model = {
                MODEL.MODEL: serialized_model,
                MODEL.ALLOW_DOWNLOAD: int(allow_download),
                MODEL.ALLOW_REMOTE_INFERENCE: int(allow_remote_inference),
                MODEL.MPC: int(mpc),
            }

            # Save serialized model into db
            # Format: { hash(worker_id + model_id) : dict( serialized_model, allow_download, allow_inference, mpc) }
            result = db_instance().hmset(key, model)

            primary_key = self._generate_hash_key()

            # Save model id
            db_instance().lpush(primary_key, model_id)

        self.cache.save(
            serialized_model,
            model_id,
            allow_download,
            allow_remote_inference,
            mpc,
            serialized=True,
        )

    def get(self, model_id: str) -> Union[Dict, None]:
        """ Retrieves model from cache/database by model_id.
            If persistent mode isn't enable, this model will be searched only in the cache memory.
            Args:
                model_id: Model's ID.
            Returns:
                result : Dict Model properties or None it not found.
        """
        if self.cache.contains(model_id):
            return self.cache.get(model_id)

        # If persistence mode enabled
        if db_instance():
            key = self._generate_hash_key(model_id)
            raw_data = db_instance().hgetall(key)

            # Decode binary keys
            raw_data = {key.decode("utf-8"): value for key, value in raw_data.items()}

            # Decode binary values
            raw_data[MODEL.ALLOW_DOWNLOAD] = bool(
                int(raw_data[MODEL.ALLOW_DOWNLOAD].decode("utf-8"))
            )
            raw_data[MODEL.ALLOW_REMOTE_INFERENCE] = bool(
                int(raw_data[MODEL.ALLOW_REMOTE_INFERENCE].decode("utf-8"))
            )
            raw_data[MODEL.MPC] = bool(int(raw_data[MODEL.MPC].decode("utf-8")))

            # Save model in cache
            self.cache.save(
                raw_data[MODEL.MODEL],
                model_id,
                raw_data[MODEL.ALLOW_DOWNLOAD],
                raw_data[MODEL.ALLOW_REMOTE_INFERENCE],
                raw_data[MODEL.MPC],
                True,
            )

            return self.cache.get(model_id)
        else:
            return None

    def remove(self, model_id: str) -> bool:
        """ Remove the specific model from cache/database.
            Args:
                model_id: Model's ID
            Returns:
                result: True if it was removed, otherwise returns False.
        """
        # Remove model from cache
        self.cache.remove(model_id)

        if db_instance():
            # Remove model ID from id's list
            ids_list_key = self._generate_hash_key()
            db_instance().lrem(ids_list_key, 0, model_id)

            # Remove model from database
            key = self._generate_hash_key(model_id)
            return db_instance().delete(key)
        else:
            return True

    def contains(self, model_id: str) -> bool:
        """ Verify if this storage instance contains the desired model.
            Args:
                model_id: Model's ID.
            Returns:
                result: True if contains, otherwise returns False.
        """
        key = self._generate_hash_key(model_id)
        if not db_instance():
            return self.cache.contains(model_id)
        else:
            return self.cache.contains(model_id) or bool(db_instance().hgetall(key))

    def _generate_hash_key(self, primary_key: str = "") -> str:
        """ To improve performance our queries will be made by hashkeys generated by 
            the aggregation between storage's id and primary key.
            Args:
                primary_key: Key/ID used to map an object.
            Returns:
                hashcode: Generated hashcode.
        """
        return hashlib.sha256(bytes(self.id + primary_key, "utf-8")).hexdigest()
