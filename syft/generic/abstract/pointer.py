from abc import abstractmethod
from typing import List
from typing import Union

from syft.serde.syft_serializable import SyftSerializable
from syft.generic.abstract.sendable import AbstractSendable
from syft.workers.abstract import AbstractWorker


class AbstractPointer(AbstractSendable, SyftSerializable):
    """A pointer to a remote object."""

    def __init__(
        self,
        location: AbstractWorker = None,
        id_at_location: Union[str, int] = None,
        owner: AbstractWorker = None,
        id: Union[str, int] = None,
        garbage_collect_data: bool = True,
        point_to_attr: str = None,
        tags: List[str] = None,
        description: str = None,
    ):
        """Initializes a pointer

        Args:
            location: An optional AbstractWorker object which points to the worker
                on which this pointer's object can be found.
            id_at_location: An optional string or integer id of the object
                being pointed to.
            owner: An optional AbstractWorker object to specify the worker on which
                the pointer is located. It is also where the pointer is
                registered if register is set to True. Note that this is
                different from the location parameter that specifies where the
                pointer points to.
            id: An optional string or integer id of the pointer.
            garbage_collect_data: If true (default), delete the remote object when the
                pointer is deleted.
            point_to_attr: string which can tell a pointer to not point directly to\
                an object, but to point to an attribute of that object such as .child or
                .grad. Note the string can be a chain (i.e., .child.child.child or
                .grad.child.child). Defaults to None, which means don't point to any attr,
                just point to then object corresponding to the id_at_location.
        """
        super().__init__(id=id, owner=owner, tags=tags, description=description)

        self.location = location
        self.id_at_location = id_at_location
        self.garbage_collect_data = garbage_collect_data
        self.point_to_attr = point_to_attr

    @abstractmethod
    def get(self, *args, **kwargs):
        pass

    @abstractmethod
    def attr(self, *args, **kwargs):
        pass

    @abstractmethod
    def setattr(self, *args, **kwargs):
        pass
