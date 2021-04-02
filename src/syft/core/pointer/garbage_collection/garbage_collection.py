"""The GC class that would handle what happens when a pointer gets deleted."""
# stdlib
from typing import Any
from typing import Dict
from typing import List

# syft relative
from ..pointer import Pointer
from .gc_strategy import GCStrategy


class GarbageCollection:
    def __init__(
        self, gc_strategy_name: str, *args: List[Any], **kwargs: Dict[Any, Any]
    ) -> None:
        if gc_strategy_name not in GCStrategy.REGISTERED_GC_STRATEGIES:
            raise ValueError(f"{gc_strategy_name} not registered!")

        gc_strategy_init = GCStrategy.REGISTERED_GC_STRATEGIES[gc_strategy_name]
        self._gc_strategy = gc_strategy_init(*args, **kwargs)

    def apply(self, pointer: Pointer) -> None:
        """Apply the GCStrategy on the pointer."""
        self._gc_strategy.reap(pointer)

    @property
    def gc_strategy(self) -> GCStrategy:
        """Get the GCStrategy."""
        return self._gc_strategy

    @gc_strategy.setter
    def gc_strategy(self, strategy: GCStrategy) -> None:
        """Set the GCStrategy."""
        self._gc_strategy = strategy
