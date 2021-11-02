# third party
import gevent

# relative
from .....logger import critical
from ....common.uid import UID
from ....store.storeable_object import StorableObject
from ...abstract.node import AbstractNode


def retrieve_object(
    node: AbstractNode, id_at_location: UID, path: str
) -> StorableObject:
    # A Soft time limit is set on celery worker which prevents infinite execution.
    while True:
        store_obj = node.store.get_object(key=id_at_location)
        if store_obj is None:
            critical(
                f"execute_action on {path} failed due to missing object"
                + f" at: {id_at_location}"
            )
            # Implicit context switch between greenlets.
            gevent.sleep(0)
        else:
            return store_obj
