import syft as sy
from syft.frameworks.torch.pointers import object_pointer

from typing import List
from typing import Union
from typing import TYPE_CHECKING

# this if statement avoids circular imports between base.py and pointer.py
if TYPE_CHECKING:
    from syft.workers import BaseWorker


class CallablePointer(object_pointer.ObjectPointer):
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


def create_callable_pointer(
    location,  #: BaseWorker,
    id: (str or int),
    id_at_location: (str or int),
    owner,  #: BaseWorker,
    tags,
    description,
    garbage_collect_data: bool = True,
    register_pointer: bool = True,
) -> object_pointer.ObjectPointer:
    """Creates a callable pointer to the object, identified by the pair (location, id_at_location).

    Note, that there is no check whether an object with this id exists at the location.
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
