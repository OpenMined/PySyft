from collections import defaultdict
from typing import Union

from syft.exceptions import ObjectNotFoundError
from syft.generic.frameworks.types import FrameworkTensor
from syft.generic.frameworks.types import FrameworkTensorType
from syft.generic.abstract.tensor import AbstractTensor
from syft.workers.abstract import AbstractWorker


class ObjectStore:
    """A storage of objects identifiable by their id.

    A wrapper object to a collection of objects where all objects
    are stored using their IDs as keys.
    """

    def __init__(self, owner: AbstractWorker = None):
        self.owner = owner

        # This is the collection of objects being stored.
        self._objects = {}
        # This is an index to retrieve objects from their tags in an efficient way
        self._tag_to_object_ids = defaultdict(set)

        # Garbage collect all remote data on a worker every garbage_delay seconds
        self.garbage_delay = 0
        # Store at most trash_capacity elements before garbage collecting
        self.trash_capacity = 10_000
        # Trash is a dict referencing for each worker key a tuple with the timestamp
        # of the last GC and the list of object to GC
        self.trash = {}

    @property
    def _tensors(self):
        return {id_: obj for id_, obj in self._objects.items() if isinstance(obj, FrameworkTensor)}

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
                raise ObjectNotFoundError(obj_id, self)
            else:
                raise e

        return obj

    def set_obj(self, obj: Union[FrameworkTensorType, AbstractTensor]) -> None:
        """Adds an object to the registry of objects.

        Args:
            obj: A torch or syft tensor with an id.
        """
        obj.owner = self.owner
        self._objects[obj.id] = obj
        # Add entry in the tag index
        if obj.tags:
            for tag in obj.tags:
                if tag not in self._tag_to_object_ids:
                    self._tag_to_object_ids[tag] = {obj.id}
                else:
                    self._tag_to_object_ids[tag].add(obj.id)

    def rm_obj(self, obj_id: Union[str, int], force=False):
        """Removes an object.

        Remove the object from the permanent object registry if it exists.

        Args:
            obj_id: A string or integer representing id of the object to be
                removed.
            force: if true, explicitly forces removal of the object modifying the
                `garbage_collect_data` attribute.
        """
        if obj_id in self._objects:
            obj = self._objects[obj_id]
            # update tag index
            if obj.tags:
                for tag in obj.tags:
                    if tag not in self._tag_to_object_ids:
                        self._tag_to_object_ids[tag].remove(obj.id)

            if force and hasattr(obj, "child") and hasattr(obj.child, "garbage_collect_data"):
                obj.child.garbage_collect_data = True

            del self._objects[obj_id]

    def force_rm_obj(self, obj_id: Union[str, int]):
        self.rm_obj(obj_id, force=True)

    def clear_objects(self):
        """Removes all objects from the object storage."""
        self._objects.clear()

    def current_objects(self):
        """Returns a copy of the objects in the object storage."""
        return self._objects.copy()

    def find_by_id(self, id):
        """Local search by id"""
        return self._objects.get(id)

    def find_by_tag(self, tag):
        """Local search by tag

        Args:
            tag (str): exact tag searched

        Return:
            A list of results, possibly empty
        """
        if tag in self._tag_to_object_ids:
            results = []
            for obj_id in self._tag_to_object_ids[tag]:
                obj = self.find_by_id(obj_id)
                if obj is not None:
                    results.append(obj)
            return results
        return []

    def register_tags(self, obj):
        # NOTE: this is a fix to correct faulty registration that can sometimes happen
        if obj.id not in self._objects:
            self.owner.register_obj(obj)

        for tag in obj.tags:
            self._tag_to_object_ids[tag].add(obj.id)

    def __len__(self):
        """
        Return the number of objects in the store
        """
        return len(self._objects)
