import syft as sy
from syft.frameworks.torch.pointers import generic_pointer
from syft import exceptions
import torch

from typing import List
from typing import Union
from typing import TYPE_CHECKING

# this if statement avoids circular imports between base.py and pointer.py
if TYPE_CHECKING:
    from syft.workers import BaseWorker


class CallablePointer(generic_pointer.GenericPointer):
    """ A class of pointers that are callable
    """

    def __init__(
        self,
        location: "BaseWorker" = None,
        id_at_location: Union[str, int] = None,
        owner: "BaseWorker" = None,
        id: Union[str, int] = None,
        garbage_collect_data: bool = True,
        point_to_attr: str = None,
        tags: List[str] = None,
        description: str = None,
    ):
        super().__init__(
            location=location,
            id_at_location=id_at_location,
            owner=owner,
            id=id,
            garbage_collect_data=garbage_collect_data,
            point_to_attr=point_to_attr,
            tags=tags,
            description=description,
        )

    def __call__(self, *args, **kwargs):
        command = ("__call__", self.id_at_location, args, kwargs)

        return_ids = [sy.ID_PROVIDER.pop()]
        response = self.owner.send_command(
            message=command, recipient=self.location, return_ids=return_ids
        )
        return response

    def get(self, deregister_ptr: bool = True):
        """Requests the object being pointed to, be serialized and return

        Note:
            This will typically mean that the remote object will be
            removed/destroyed.

        Args:
            deregister_ptr (bool, optional): this determines whether to
                deregister this pointer from the pointer's owner during this
                method. This defaults to True because the main reason people use
                this method is to move the tensor from the remote machine to the
                local one, at which time the pointer has no use.

        Returns:
            An AbstractObject object which is the tensor (or chain) that this
            object used to point to on a remote machine.

        TODO: add param get_copy which doesn't destroy remote if true.
        """

        if self.point_to_attr is not None:

            raise exceptions.CannotRequestObjectAttribute(
                "You called .get() on a pointer to"
                " a tensor attribute. This is not yet"
                " supported. Call .clone().get() instead."
            )

        # if the pointer happens to be pointing to a local object,
        # just return that object (this is an edge case)
        if self.location == self.owner:
            obj = self.owner.get_obj(self.id_at_location).child
        else:
            # get tensor from remote machine
            obj = self.owner.request_obj(self.id_at_location, self.location)

        # Register the result
        assigned_id = self.id_at_location
        self.owner.register_obj(obj, assigned_id)

        # Remove this pointer by default
        if deregister_ptr:
            self.owner.de_register_obj(self)

        return obj


def create_callable_pointer(
    location,  #: BaseWorker,
    id: (str or int),
    id_at_location: (str or int),
    owner,  #: BaseWorker,
    tags,
    description,
    garbage_collect_data: bool = True,
    register_pointer: bool = True,
) -> generic_pointer.GenericPointer:
    """Creates a pointer to the "obj"
    """

    if id is None:
        id = sy.ID_PROVIDER.pop()

    ptr = CallablePointer(
        location=location,
        id_at_location=id_at_location,
        owner=owner,
        id=id,
        garbage_collect_data=garbage_collect_data,
        tags=tags,
        description=description,
    )

    if register_pointer:
        owner.register_obj(ptr)

    return ptr
