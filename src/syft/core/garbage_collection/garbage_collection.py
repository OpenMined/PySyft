"""The GC class that would handle what happens when a pointer gets deleted."""

# syft relative
from ..pointer.pointer import Pointer
from .gc_strategy import GCStrategy


class GarbageCollection:
    def __init__(self, gc_strategy: GCStrategy) -> None:
        self._gc_strategy = gc_strategy

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
