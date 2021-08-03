# stdlib
from typing import Any
from typing import Dict
from typing import List


class ScalarChainManagerTensor:
    """Supports convenience methods for scalar chains of abstraction"""

    def __init__(self) -> None:
        pass

    def push_abstraction_top(
        self, scalar_type: Any, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> None:
        """ """


class TensorChainManager:
    def __init__(self) -> None:
        pass

    def push_abstraction_top(
        self, tensor_type: Any, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> None:
        """ """
        self.child: Any = tensor_type(self.child, *args, **kwargs)

    def replace_abstraction_top(
        self, tensor_type: Any, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> None:
        self.child = tensor_type(*args, **kwargs)
