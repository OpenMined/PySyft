# """The GC class that would handle what happens when a pointer gets deleted."""
# # stdlib
# from typing import Any

# # third party
# from typing_extensions import final

# # relative
# from ..pointer import Pointer
# from .gc_strategy import GCStrategy


# @final
# class GarbageCollection:
#     __slots__ = ["_gc_strategy"]

#     _gc_strategy: GCStrategy

#     def __init__(self, gc_strategy_name: str, *args: Any, **kwargs: Any) -> None:
#         """Initialize a strategy given the name of it.

#         Args:
#             gc_strategy_name (str): name of the registered GC strategy
#             *args (List[Any]): args to be passed to the GCStrategy init method
#             **kwargs (Dict[Any, Any]): kwargs to be passed to the GCStrategy
#                             init method
#         Return:
#             None
#         """
#         if gc_strategy_name not in GCStrategy.REGISTERED_GC_STRATEGIES:
#             raise ValueError(f"{gc_strategy_name} not registered!")

#         gc_strategy_init = GCStrategy.REGISTERED_GC_STRATEGIES[gc_strategy_name]
#         self._gc_strategy = gc_strategy_init(*args, **kwargs)

#     def apply(self, pointer: Pointer) -> None:
#         """Apply the GCStrategy garbage collection logic on the pointer.

#         Args:
#             pointer (Pointer): the pointer that should be garbage collected

#         Return:
#             None
#         """
#         self._gc_strategy.reap(pointer)

#     @property
#     def gc_strategy(self) -> GCStrategy:
#         """Get the GCStrategy.

#         Return:
#             The used GCStrategy
#         """
#         return self._gc_strategy

#     @gc_strategy.setter
#     def gc_strategy(self, strategy: GCStrategy) -> None:
#         """Set the GCStrategy.

#         Args:
#             strategy (GCStrategy): The GCStrategy to be set

#         Return:
#             None
#         """
#         self._gc_strategy = strategy
