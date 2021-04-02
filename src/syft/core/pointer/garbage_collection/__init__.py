"""Strategies that clients could use to trigger the garbage collection process."""
# syft relative
from .garbage_collection import GarbageCollection
from .gc_batched import GCBatched
from .gc_simple import GCSimple
from .gc_strategy import GCStrategy

__GC_DEFAULT_STRATEGY: str = "gcsimple"


def gc_get_default_strategy() -> str:
    return __GC_DEFAULT_STRATEGY


def gc_set_default_strategy(gc_strategy_name: str) -> None:
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
