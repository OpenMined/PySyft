from typing import List
from typing import Union
from typing import TYPE_CHECKING

import syft as sy
from syft.generic.pointers.callable_pointer import create_callable_pointer
from syft.workers.abstract import AbstractWorker
from syft.generic.abstract.syft_serializable import SyftSerializable

# this if statement avoids circular imports between base.py and pointer.py
if TYPE_CHECKING:
    from syft.workers.base import BaseWorker


class ObjectWrapper(SyftSerializable):
    """A class that wraps an arbitrary object and provides it with an id, tags, and description"""

    def __init__(self, obj, id: int, tags: List[str] = None, description: str = None):
        """object wrapper initialization
        Args:
            obj: An arbitrary object, can also be a function
            id: id to be associated with the object
            tags: list of strings of tags of the object
            description: a description of the object
        """
        self._obj = obj
        self.id = id
        self.tags = tags
        self.description = description

    def __call__(self, *args, **kwargs):
        return self._obj(*args, **kwargs)

    def __str__(self):
        out = "<"
        out += str(type(self)).split("'")[1].split(".")[-1]
        out += " id:" + str(self.id)
        out += " obj:" + str(self._obj)
        out += ">"
        return out

    def __repr__(self):
        return str(self)

    @property
    def obj(self):
        return self._obj

    @staticmethod
    def create_pointer(
        object,
        owner: "BaseWorker",
        location: "BaseWorker",
        ptr_id: Union[int, str],
        id_at_location: Union[int, str] = None,
        garbage_collect_data=None,
        **kwargs,
    ):
        """Creates a callable pointer to the object wrapper instance

        Args:
            owner: A BaseWorker parameter to specify the worker on which the
                pointer is located. It is also where the pointer is registered
                if register is set to True.
            location: The BaseWorker object which points to the worker on which
                this pointer's object can be found. In nearly all cases, this
                is self.owner and so this attribute can usually be left blank.
                Very rarely you may know that you are about to move the Tensor
                to another worker so you can pre-initialize the location
                attribute of the pointer to some other worker, but this is a
                rare exception.
            ptr_id: A string or integer parameter to specify the id of the pointer.
            id_at_location: A string or integer id of the object being pointed
                to. Similar to location, this parameter is almost always
                self.id and so you can leave this parameter to None. The only
                exception is if you happen to know that the ID is going to be
                something different than self.id, but again this is very rare
                and most of the time, setting this means that you are probably
                doing something you shouldn't.
            garbage_collect_data: If True, delete the remote object when the
                pointer is deleted.

        Returns:
            A pointers.CallablePointer pointer to self.
        """
        pointer = create_callable_pointer(
            owner=owner,
            location=location,
            id=ptr_id,
            id_at_location=id_at_location if id_at_location is not None else object.id,
            tags=object.tags,
            description=object.description,
            garbage_collect_data=False if garbage_collect_data is None else garbage_collect_data,
        )
        return pointer

    @staticmethod
    def simplify(worker: AbstractWorker, obj: "ObjectWrapper") -> tuple:
        return (obj.id, sy.serde.msgpack.serde._simplify(worker, obj.obj))

    @staticmethod
    def detail(worker: AbstractWorker, obj_wrapper_tuple: str) -> "ObjectWrapper":
        obj_wrapper = ObjectWrapper(
            id=obj_wrapper_tuple[0],
            obj=sy.serde.msgpack.serde._detail(worker, obj_wrapper_tuple[1]),
        )
        return obj_wrapper
