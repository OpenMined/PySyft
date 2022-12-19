# """The strategy that all GC Heuristics should implement."""
# # stdlib
# from abc import ABC
# from abc import abstractmethod
# from typing import Dict
# from typing import Type

# # relative
# from ..pointer import Pointer


# class GCStrategy(ABC):
#     """The Strategy that all GC Heuristics should inherit."""

#     REGISTERED_GC_STRATEGIES: Dict[str, Type["GCStrategy"]] = {}

#     @staticmethod
#     def _register(cls: Type["GCStrategy"]) -> None:
#         """Add a concrete implementation for the GC Strategy to a dictionary
#         of known strategies.

#         This is used to have a simpler way when we initialize the GarbageCollection
#         class.

#         Args:
#             cls (Type[GCStrategy]): The subclass that inherits from GCStrategy

#         Return:
#             None
#         """
#         gc_strategy_name = cls.__name__.lower()
#         if gc_strategy_name in GCStrategy.REGISTERED_GC_STRATEGIES:
#             raise ValueError(f"{gc_strategy_name} already registered!")

#         GCStrategy.REGISTERED_GC_STRATEGIES[gc_strategy_name] = cls

#     def __init_subclass__(cls: Type["GCStrategy"]) -> None:
#         """Called when a class inherits the GCStrategy.

#         Args:
#             cls (Type[GCStrategy]): The subclass that inherits from GCStrategy

#         Return:
#             None
#         """
#         super().__init_subclass__()
#         GCStrategy._register(cls)

#     @abstractmethod
#     def reap(self, pointer: Pointer) -> None:
#         """What happens when a Ponter is deleted.
#         It should be implemented in the GCStrategy that extends this class
#         """
#         pass
