from flask import Blueprint

import syft as sy
import torch as th

# Global variables must be initialized here.
hook = sy.TorchHook(th)
local_worker = hook.local_worker
hook.local_worker.is_client_worker = False

html = Blueprint(r"html", __name__)
ws = Blueprint(r"ws", __name__)


from . import routes, events
from .persistence.models import db


# Implement base search locally
# We need this local fix for now to be able run the search operation on Grid
# TODO: remove this after this issue is fixed https://github.com/OpenMined/PySyft/issues/2609

from syft.generic.frameworks.types import FrameworkTensor


def _search(self, *query):
    """Search for a match between the query terms and a tensor's Id, Tag, or Description.
https://github.com/OpenMined/PySyft/issues/2609
    Note that the query is an AND query meaning that every item in the list of strings (query*)
    must be found somewhere on the tensor in order for it to be included in the results.

    Args:
        query: A list of strings to match against.
        me: A reference to the worker calling the search.

    Returns:
        A list of PointerTensors.
    """
    results = list()
    for key, obj in self._objects.items():
        found_something = True
        for query_item in query:
            # If deserialization produced a bytes object instead of a string,
            # make sure it's turned back to a string or a fair comparison.
            if isinstance(query_item, bytes):
                query_item = query_item.decode("ascii")

            match = False
            if query_item == str(key):
                match = True

            if isinstance(obj, FrameworkTensor):
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
            ptr = obj.create_pointer(garbage_collect_data=False, owner=sy.local_worker)
            results.append(ptr)

    return results


sy.workers.base.BaseWorker.search = _search
