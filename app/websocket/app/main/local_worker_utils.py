"""Local worker utilities."""

from . import local_worker


def register_obj(obj, obj_id=None):
    """Registers the specified object with the local worker.

    Args:
        obj: A torch Tensor or Variable object to be registered.
        obj_id (int or string): random integer between 0 and 1e10 or
        string uniquely identifying the object.
    """
    if obj_id is None:
        obj_id = obj.id

    local_worker._objects[obj_id] = obj


def get_obj(obj_id):
    """Get object from local worker."""
    return local_worker.get_obj(obj_id)


def get_objs():
    """Get all objects from local worker."""
    return local_worker._objects
