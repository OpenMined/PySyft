# stdlib
from typing import Tuple

# third party
import gevent

# relative
from .....logger import critical
from ....common.uid import UID
from ....store.storeable_object import StorableObject
from ...abstract.node import AbstractNode


def retrieve_object(
    node: AbstractNode, id_at_location: UID, path: str, proxy_only: bool = False
) -> StorableObject:
    # A hard time limit is set on celery worker which prevents infinite execution.
    ctr = 0
    while True:
        store_obj = node.store.get_or_none(key=id_at_location, proxy_only=proxy_only)
        if store_obj is None:
            if ctr % 1500 == 0:
                critical(
                    f"execute_action on {path} failed due to missing object"
                    + f" at: {id_at_location}"
                )
            # Implicit context switch between greenlets.
            gevent.sleep(0)
            ctr += 1
        else:
            return store_obj


def beaver_retrieve_object(
    node: AbstractNode, id_at_location: UID, nr_parties: int
) -> StorableObject:
    # relative
    from .beaver_action import BEAVER_CACHE

    # A hard time limit is set on celery worker which prevents infinite execution.
    ctr = 0
    while True:
        store_obj = BEAVER_CACHE.get(id_at_location, None)  # type: ignore
        if store_obj is None or len(store_obj.data) != nr_parties:
            if ctr % 1500 == 0:
                critical(
                    f"Beaver Retrieval failed for {nr_parties} parties due to missing object"
                    + f" at: {id_at_location} values: {store_obj}"
                )
            # Implicit context switch between greenlets.
            gevent.sleep(0)
            ctr += 1
        else:
            return store_obj


def crypto_store_retrieve_object(  # type: ignore
    op_str: str,
    **kwargs,
) -> Tuple:
    # relative
    from ....smpc.store.exceptions import EmptyPrimitiveStore
    from ....tensor.smpc.share_tensor import ShareTensor

    crypto_store = ShareTensor.crypto_store

    # A hard time limit is set on celery worker which prevents infinite execution.
    ctr = 0
    while True:
        try:
            store_values = crypto_store.get_primitives_from_store(
                op_str, **kwargs  # type: ignore
            )
            return tuple(store_values)
        except EmptyPrimitiveStore:

            if ctr % 1500 == 0:
                critical(
                    f"Crypto Store Retrieval failed for parties due to missing object: {EmptyPrimitiveStore}"
                )
            # Implicit context switch between greenlets.
            gevent.sleep(0)
            ctr += 1
