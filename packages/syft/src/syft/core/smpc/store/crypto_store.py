"""CryptoStore manages the needed crypto primitives."""

# stdlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List

# relative
from ...common.serde.serializable import serializable


@serializable(recursive_serde=True)
class CryptoStore:
    """Manages items needed for MPC Computation.

    Attributes:
        store (Dict[Any, Any]): Object needed for MPC.

    """

    __slots__ = ("store",)
    __attr_allowlist__ = ("store",)

    _func_add_store: Dict[Any, Callable] = {}
    _func_get_store: Dict[Any, Callable] = {}

    def __init__(self) -> None:
        """Initializer."""
        self.store: Dict[Any, Any] = {}

    def populate_store(
        self,
        op_str: str,
        primitives: Iterable[Any],
        *args: List[Any],
        **kwargs: Dict[Any, Any]
    ) -> None:
        """Populate items.

        Args:
            op_str (str): Operator to store.
            primitives (Iterables[Any]): Primitives to store.
            *args: Variable length arguments.
            **kwargs: Keywords arguments.
        """
        populate_func = CryptoStore._func_add_store[op_str]
        populate_func(self.store, primitives, *args, **kwargs)

    def get_primitives_from_store(
        self, op_str: str, ring_size: int, *args: List[Any], **kwargs: Dict[Any, Any]
    ) -> List[Any]:
        """Get primitives from store.

        Args:
            op_str (str): Operator to get.
            nr_instances (int): Number of instances.
            *args: Variable length arguments.
            **kwargs: Keywords arguments.

        Returns:
            Dict[Any, Any]: Primitives.
        """
        retrieve_func = CryptoStore._func_get_store[op_str]
        primitives = retrieve_func(
            store=self.store, ring_size=ring_size, *args, **kwargs
        )

        return primitives
