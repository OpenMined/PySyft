# stdlib
from typing import Any

# relative
from .passthrough import SupportedChainType  # type: ignore


class ScalarChainManagerTensor:
    """Supports convenience methods for scalar chains of abstraction"""

    def __init__(self) -> None:
        pass

    def push_abstraction_top(self, scalar_type: Any, *args: Any, **kwargs: Any) -> None:
        """ """


class TensorChainManager:
    def __init__(self, child: SupportedChainType) -> None:
        self.child = child

    def push_abstraction_top(self, tensor_type: Any, *args: Any, **kwargs: Any) -> None:
        """ """
        self.child = tensor_type(self.child, *args, **kwargs)

    def replace_abstraction_top(
        self, tensor_type: Any, *args: Any, **kwargs: Any
    ) -> None:
        self.child = tensor_type(*args, **kwargs)
