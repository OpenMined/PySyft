# Standard Python imports
from typing import Union

from syft.exceptions import ObjectNotFoundError
from syft.generic.abstract.tensor import AbstractTensor
from syft.generic.frameworks.types import FrameworkTensorType
from syft.generic.object_storage import ObjectStore

# External imports
from syft.serde import deserialize, serialize
from syft.workers.base import BaseWorker

# Local imports
from .database import db_instance


def set_persistent_mode(redis_db):
    """Update/Overwrite PySyft ObjectStore to work in a persistent mode.

    Args:
        redis_db : Redis database instance.
    """

    # Updated methods

    def _set_obj(self, obj: Union[FrameworkTensorType, AbstractTensor]) -> None:
        self._objects[obj.id] = obj
        redis_db.hset(self.id, obj.id, serialize(obj))

    def _rm_obj(self, remote_key: Union[str, int]):
        if remote_key in self._objects:
            del self._objects[remote_key]
        redis_db.hdel(self.id, remote_key)

    def _force_rm_obj(self, remote_key: Union[str, int]):
        if remote_key in self._objects:
            obj = self._objects[remote_key]
            if hasattr(obj, "child") and hasattr(obj.child, "garbage_collect_data"):
                obj.child.garbage_collect_data = True
            del self._objects[remote_key]
        redis_db.hdel(self.id, remote_key)

    def _get_obj(self, obj_id: Union[str, int]) -> object:
        try:
            obj = self._objects[obj_id]
        except KeyError as e:
            if obj_id not in self._objects:
                # Try to recover object on database
                obj = redis_db.hget(self.id, obj_id)
                if obj:
                    self._objects[obj_id] = deserialize(obj)
                else:
                    raise ObjectNotFoundError(obj_id, self)
            else:
                raise e
        return obj

    # Overwrite default object storage methods
    ObjectStore.set_obj = _set_obj
    ObjectStore.get_obj = _get_obj
    ObjectStore.rm_obj = _rm_obj
    ObjectStore.force_rm_obj = _force_rm_obj


def recover_objects(worker) -> BaseWorker:
    """Retrieves all database objects for a given worker.

    Args:
        worker : Worker Instance.
    Returns:
        worker: Updated worker instance.
    """
    if db_instance():
        raw_objs = db_instance().hgetall(worker.id)
        objects = {
            int(key.decode("utf-8")): deserialize(value)
            for key, value in raw_objs.items()
        }
        worker._objects = objects
    return worker
