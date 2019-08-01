from typing import List
from typing import Union

import torch

from syft.frameworks.torch.tensors.interpreters import AbstractTensor


class ObjectStorage:
    """A storage of objects identifiable by their id.

    A wrapper object to a collection of objects where all objects
    are stored using their IDs as keys.
    """

    def __init__(self):
        self._objects = {}

    def register_obj(self, obj: object, obj_id: Union[str, int] = None):
        """Registers the specified object with the current worker node.

        Selects an id for the object, assigns a list of owners, and establishes
        whether it's a pointer or not. This method is generally not used by the
        client and is instead used by internal processes (hooks and workers).

        Args:
            obj: A torch Tensor or Variable object to be registered.
            obj_id (int or string): random integer between 0 and 1e10 or
            string uniquely identifying the object.
        """
        if obj_id is not None and hasattr(obj, "id"):
            obj.id = obj_id
        self.set_obj(obj)

    def de_register_obj(self, obj: object, _recurse_torch_objs: bool = True):
        """Deregisters the specified object.

        Deregister and remove attributes which are indicative of registration.

        Args:
            obj: A torch Tensor or Variable object to be deregistered.
            _recurse_torch_objs: A boolean indicating whether the object is
                more complex and needs to be explored. Is not supported at the
                moment.
        """
        if hasattr(obj, "id"):
            self.rm_obj(obj.id)
        if hasattr(obj, "_owner"):
            del obj._owner

    def get_obj(self, obj_id: Union[str, int]) -> object:
        """Returns the object from registry.

        Look up an object from the registry using its ID.

        Args:
            obj_id: A string or integer id of an object to look up.

        Returns:
            Object with id equals to `obj_id`.
        """

        try:
            obj = self._objects[obj_id]
        except KeyError as e:
            if obj_id not in self._objects:
                msg = 'Object "' + str(obj_id) + '" not found on worker!!!'
                msg += (
                    "You just tried to interact with an object ID:"
                    + str(obj_id)
                    + " on "
                    + str(self)
                    + " which does not exist!!! "
                )
                msg += (
                    "Use .send() and .get() on all your tensors to make sure they're"
                    "on the same machines. "
                    "If you think this tensor does exist, check the ._objects dictionary"
                    "on the worker and see for yourself!!! "
                    "The most common reason this error happens is because someone calls"
                    ".get() on the object's pointer without realizing it (which deletes "
                    "the remote object and sends it to the pointer). Check your code to "
                    "make sure you haven't already called .get() on this pointer!!!"
                )
                raise KeyError(msg)
            else:
                raise e

        return obj

    def set_obj(self, obj: Union[torch.Tensor, AbstractTensor]) -> None:
        """Adds an object to the registry of objects.

        Args:
            obj: A torch or syft tensor with an id.
        """
        self._objects[obj.id] = obj

    def rm_obj(self, remote_key: Union[str, int]):
        """Removes an object.

        Remove the object from the permanent object registry if it exists.

        Args:
            remote_key: A string or integer representing id of the object to be
                removed.
        """
        if remote_key in self._objects:
            del self._objects[remote_key]

    def force_rm_obj(self, remote_key: Union[str, int]):
        """Forces object removal.

        Besides removing the object from the permanent object registry if it exists.
        Explicitly forces removal of the object modifying the `garbage_collect_data` attribute.

        Args:
            remote_key: A string or integer representing id of the object to be
                removed.
        """
        if remote_key in self._objects:
            obj = self._objects[remote_key]
            if hasattr(obj, "child"):
                obj.child.garbage_collect_data = True
            del self._objects[remote_key]

    def clear_objects(self, return_self: bool = True):
        """Removes all objects from the object storage.

        Note: the "return self" statement is kept in order to avoid modifying the code shown in the udacity course.

        Args:
            return_self: flag, whether to return self as return value

        Returns:
            self, if return_self if True, else None

        """
        self._objects.clear()
        return self if return_self else None

    def current_objects(self):
        """Returns a copy of the objects in the object storage."""
        return self._objects.copy()
