from flask import Blueprint

import syft as sy
import torch as th

from typing import List
from typing import Tuple
from typing import Union

from syft.serde import serialize, deserialize
from syft.generic.frameworks.types import FrameworkTensor
from syft.generic.object import AbstractObject
from syft.generic.pointers.pointer_tensor import PointerTensor


def _search(self, query: Union[List[Union[str, int]], str, int]) -> List[PointerTensor]:
    """Search for a match between the query terms and a tensor's Id, Tag, or Description.
    Note that the query is an AND query meaning that every item in the list of strings (query*)
    must be found somewhere on the tensor in order for it to be included in the results.
    Args:
        query: A list of strings to match against.
        me: A reference to the worker calling the search.
    Returns:
        A list of PointerTensors.
    """
    if isinstance(query, (str, int)):
        query = [query]

    results = list()
    for key, obj in self._objects.items():
        found_something = True
        for query_item in query:
            # If deserialization produced a bytes object instead of a string,
            # make sure it's turned back to a string or a fair comparison.
            if isinstance(query_item, bytes):
                query_item = query_item.decode("ascii")
            query_item = str(query_item)

            match = False
            if query_item == str(key):
                match = True

            if isinstance(obj, (AbstractObject, FrameworkTensor)):
                if obj.tags is not None:
                    if query_item in obj.tags:
                        match = True

                if obj.description is not None:
                    if query_item in obj.description:
                        match = True

            if not match:
                found_something = False
        if found_something:
            # set garbage_collect_data to False because if we're searching
            # for a tensor we don't own, then it's probably someone else's
            # decision to decide when to delete the tensor.
            ptr = obj.create_pointer(
                garbage_collect_data=False, owner=sy.local_worker, location=self
            ).wrap()
            results.append(ptr)

    return results


# Overwrite PySyft search method
sy.workers.base.BaseWorker.search = _search

# Global variables must be initialized here.
hook = sy.TorchHook(th)
local_worker = sy.VirtualWorker(hook, auto_add=False)
hook.local_worker.is_client_worker = False

html = Blueprint(r"html", __name__)
ws = Blueprint(r"ws", __name__)


from . import routes, events
from . import auth
