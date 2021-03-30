"""Strategies that clients could use to trigger the garbage collection process."""
# syft relative
from .garbage_collection import GarbageCollection
from .gc_batched import GCBatched
from .gc_simple import GCSimple
from .gc_strategy import GCStrategy

__GC_DEFAULT_STRATEGY: GCStrategy = GCSimple()


def gc_get_default_strategy() -> GCStrategy:
    return __GC_DEFAULT_STRATEGY


def gc_set_default_strategy(gc_strategy: GCStrategy) -> None:
    global __GC_DEFAULT_STRATEGY
    __GC_DEFAULT_STRATEGY = gc_strategy


__all__ = [
    "GarbageCollection",
    "GCSimple",
    "GCBatched",
    "get_default_strategy",
    "set_default_strategy",
]
