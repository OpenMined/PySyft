from typing import List
from typing import Union

from syft.generic.pointers.object_pointer import ObjectPointer
from syft.workers.base import BaseWorker


class StringPointer(ObjectPointer):
    # , metaclass=PointerClassMaker, pointed_type=String):
    """
       This class defines a pointer to a 'String' object that might live
    on a remote machine. In other words, it holds a pointer to a
    'String' object owned by a possibly different worker (although
    it can also point to a String owned by the same worker'.

    All String method are hooked to objects of this class, and calls to
    such methods are forwarded to the pointed-to String object.
    """

    def __init__(
        self,
        location: BaseWorker = None,
        id_at_location: Union[str, int] = None,
        owner: BaseWorker = None,
        id: Union[str, int] = None,
        garbage_collect_data: bool = True,
        tags: List[str] = None,
        description: str = None,
    ):

        super(StringPointer, self).__init__(
            location=location,
            id_at_location=id_at_location,
            owner=owner,
            id=id,
            garbage_collect_data=garbage_collect_data,
            tags=tags,
            description=description,
        )
