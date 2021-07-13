"""Utils functions that might be used into any module."""

# stdlib
import asyncio
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import functools
from itertools import repeat
import operator
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union


def ispointer(obj: Any) -> bool:
    """Check if a given obj is a pointer (is a remote object).

    Args:
        obj (Any): Object.

    Returns:
        bool: True (if pointer) or False (if not).
    """
    if type(obj).__name__.endswith("Pointer") and hasattr(obj, "id_at_location"):
        return True
    return False


def parallel_execution(
    fn: Callable[..., Any],
    parties: Union[None, List[Any]] = None,
) -> Callable[..., List[Any]]:
    # Wait 10 seconds for all the remote operations to finish
    TIMEOUT_PERIOD_S = 10

    """Wrap a function such that it can be run in parallel at multiple parties.

    Args:
        fn (Callable): The function to run.
        parties (Union[None, List[Any]]): Clients from syft. If this is set, then we extract
          the method from the AST from the client. Defaults to None.

    Returns:
        Callable[..., List[Any]]: A Callable that returns a list of results.
    """

    def initializer(event_loop) -> None:
        """Set the same event loop to other threads/processes.

        This is needed because there are new threads/processes started with
        the Executor and they do not have have an event loop set

        Args:
            event_loop: The event loop.
        """
        asyncio.set_event_loop(event_loop)

    @functools.wraps(fn)
    def wrapper(
        args: List[List[Any]],
        kwargs: Optional[Dict[Any, Dict[Any, Any]]] = None,
    ) -> List[Any]:
        """Wrap sanity checks and checks what executor should be used.

        Args:
            args (List[List[Any]]): Args.
            kwargs (Optional[Dict[Any, Dict[Any, Any]]]): Kwargs. Default to None.

        Returns:
            List[Any]: Results from the parties
        """
        # Each party has a list of args and a dictionary of kwargs
        nr_parties = len(args)

        if args is None:
            args = [[] for i in range(nr_parties)]

        if kwargs is None:
            kwargs = {}

        if parties:
            func_name = f"{fn.__module__}.{fn.__qualname__}"
            attr_getter = operator.attrgetter(func_name)
            funcs = [attr_getter(party) for party in parties]
        else:
            funcs = list(repeat(fn, nr_parties))

        map_futures_to_idx = {}
        loop = asyncio.get_event_loop()

        # Create a list with empty positions
        local_shares = list(range(nr_parties))

        with ThreadPoolExecutor(
            max_workers=nr_parties, initializer=initializer, initargs=(loop,)
        ) as executor:
            for i in range(nr_parties):
                _args = args[i]
                _kwargs = kwargs
                map_futures_to_idx[executor.submit(funcs[i], *_args, **_kwargs)] = i

        for future in as_completed(
            list(map_futures_to_idx.keys()), timeout=TIMEOUT_PERIOD_S
        ):
            idx_list = map_futures_to_idx[future]
            local_shares[idx_list] = future.result()

        return local_shares

    return wrapper
