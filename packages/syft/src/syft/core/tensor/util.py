# stdlib
import asyncio
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import functools
import operator
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Tuple as TypeTuple
from typing import Type
from typing import Union

HANDLED_FUNCTIONS: Dict[Any, Any] = {}


def inputs2child(*args: Tuple[Any, ...], **kwargs: Any) -> TypeTuple[Tuple[Any], Any]:

    # relative
    from .passthrough import PassthroughTensor  # type: ignore

    out_args_list = list()

    for out in tuple(
        [x.child if isinstance(x, PassthroughTensor) else x for x in args]
    ):
        out_args_list.append(out)

    out_kwargs = {}

    for x in kwargs.items():
        if isinstance(x[1], PassthroughTensor):
            out_kwargs[x[0]] = x[1].child
        else:
            out_kwargs[x[0]] = x[1]

    out_args = tuple(out_args_list)

    return out_args, out_kwargs  # type: ignore


def query_implementation(tensor_type: Any, func: Any) -> Any:
    name = func.__name__
    cache = HANDLED_FUNCTIONS.get(tensor_type, None)
    if cache and name in cache:
        return HANDLED_FUNCTIONS[tensor_type][func.__name__]
    return None


def implements(tensor_type: Any, np_function: Any) -> Any:
    def decorator(func: Any) -> Any:
        if tensor_type not in HANDLED_FUNCTIONS:
            HANDLED_FUNCTIONS[tensor_type] = {}

        HANDLED_FUNCTIONS[tensor_type][np_function.__name__] = func
        return func

    return decorator


def parallel_execution(
    fn: Callable[..., Any],
    parties: Union[None, List[Any]] = None,
    cpu_bound: bool = False,
) -> Callable[..., List[Any]]:
    """Wrap a function such that it can be run in parallel at multiple parties.

    Args:
        fn (Callable): The function to run.
        parties (Union[None, List[Any]]): Clients from syft. If this is set, then the
            function should be run remotely. Defaults to None.
        cpu_bound (bool): Because of the GIL (global interpreter lock) sometimes
            it makes more sense to use processes than threads if it is set then
            processes should be used since they really run in parallel if not then
            it makes sense to use threads since there is no bottleneck on the CPU side

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
        executor: Union[Type[ProcessPoolExecutor], Type[ThreadPoolExecutor]]
        if cpu_bound:
            executor = ProcessPoolExecutor
        else:
            executor = ThreadPoolExecutor

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

        futures = []
        loop = asyncio.get_event_loop()

        with executor(
            max_workers=nr_parties, initializer=initializer, initargs=(loop,)
        ) as executor:
            for i in range(nr_parties):
                _args = args[i]
                _kwargs = kwargs
                futures.append(executor.submit(funcs[i], *_args, **_kwargs))

        local_shares = [f.result() for f in futures]

        return local_shares

    return wrapper
