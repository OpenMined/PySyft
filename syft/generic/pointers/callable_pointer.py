from typing import List
from typing import Union
from typing import TYPE_CHECKING

import syft as sy
from syft.generic.pointers.object_pointer import ObjectPointer

# this if statement avoids circular imports
if TYPE_CHECKING:
    from syft.workers.base import BaseWorker


class CallablePointer(ObjectPointer):
    """A class of pointers that are callable

    A CallablePointer is an ObjectPointer which implements the __call__ function.
    This lets you execute a command directly on the object to which it points.
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
        """

        Args:
            location: An optional BaseWorker object which points to the worker
                on which this pointer's object can be found.
            id_at_location: An optional string or integer id of the object
                being pointed to.
            owner: An optional BaseWorker object to specify the worker on which
                the pointer is located. It is also where the pointer is
                registered if register is set to True. Note that this is
                different from the location parameter that specifies where the
                pointer points to.
            id: An optional string or integer id of the PointerTensor.
            garbage_collect_data: If true (default), delete the remote object when the
                pointer is deleted.
            point_to_attr: string which can tell a pointer to not point directly to\
                an object, but to point to an attribute of that object such as .child or
                .grad. Note the string can be a chain (i.e., .child.child.child or
                .grad.child.child). Defaults to None, which means don't point to any attr,
                just point to then object corresponding to the id_at_location.
        """
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

        return_ids = (sy.ID_PROVIDER.pop(),)
        response = self.owner.send_command(
            cmd_name="__call__",
            target=self.id_at_location,
            args_=args,
            kwargs_=kwargs,
            recipient=self.location,
            return_ids=return_ids,
        )
        return response


def create_callable_pointer(
    location: "BaseWorker",
    id: (str or int),
    id_at_location: (str or int),
    owner: "BaseWorker",
    tags,
    description,
    garbage_collect_data: bool = True,
    register_pointer: bool = True,
) -> ObjectPointer:
    """Creates a callable pointer to the object identified by the pair (location, id_at_location).

    Note, that there is no check whether an object with this id exists at the location.

    Args:
        location:
        id:
        id_at_location:
        owner:
        tags:
        description:
        garbage_collect_data:
        register_pointer:
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
