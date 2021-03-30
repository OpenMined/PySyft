"""The strategy that all GC Heuristics should implement."""

# stdlib
from abc import ABC
from abc import abstractmethod

# syft relative
from ..pointer.pointer import Pointer


class GCStrategy(ABC):
    """The Strategy that all GC Heuristics should inherit."""

    @abstractmethod
    def reap(self, pointer: Pointer) -> None:
        """What happens when a Ponter is deleted.
        It should be implemented in the GCStrategy that extends this class
        """
        pass
