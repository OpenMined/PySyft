"""Strategies that clients could use to trigger the garbage collection process.

For the moment, there are implemented two GC strategies: GCSimple and GCBatched
Information about each strategy could be found in the implementation files.

By default, the strategy that is utilised is the "gcsimple" one.

The GCStrategy is on a per client basis.


1. How to use a specific strategy?

    client.gc = GarbageCollection(<strategy_name>, <args>, <kwargs>)

or

    client.gc.gc_strategy = <GCStrategyInstance>


2. How to implement a new strategy?
  - Create a new implementation file in the garbage_collection folder with a
  intuitive name
  - Implement the "reap" method (mandatory!) -- what should happen when a pointer
  object is deleted
  - Implement the "__del__" method (if necessary) -- this is primary used when
  a user would change the strategy on the fly
  - Add your method to the "__init__.py" file (this file)
  - Add tests for your new GCStrategy
  - And that's all!
"""
# relative
from .garbage_collection import GarbageCollection
from .gc_batched import GCBatched
from .gc_simple import GCSimple
from .gc_strategy import GCStrategy

__GC_DEFAULT_STRATEGY: str = "gcsimple"


def gc_get_default_strategy() -> str:
    """Retrieve the default garbage collection strategy.

    Return:
        A string representing the default strategy
    """
    return __GC_DEFAULT_STRATEGY


def gc_set_default_strategy(gc_strategy_name: str) -> None:
    """Set the default garbage collection strategy for new clients.

    Args:
        gc_strategy_name (str): the garbage collection strategy name

    Return:
        None
    """
    global __GC_DEFAULT_STRATEGY
    if gc_strategy_name not in GCStrategy.REGISTERED_GC_STRATEGIES:
        raise ValueError(f"{gc_strategy_name} not registered!")
    __GC_DEFAULT_STRATEGY = gc_strategy_name


__all__ = [
    "GarbageCollection",
    "GCSimple",
    "GCBatched",
    "get_default_strategy",
    "set_default_strategy",
]
