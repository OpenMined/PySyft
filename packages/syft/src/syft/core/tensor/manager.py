# stdlib
from typing import Any
from typing import Dict
from typing import List


class ScalarChainManagerTensor:
    """Supports convenience methods for scalar chains of abstraction"""

    def push_abstraction_top(
        self, scalar_type: type, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> None:
        """ """
        pass


class TensorChainManager:
    def push_abstraction_top(
        self, tensor_type: type, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> None:
        """ """
        self.child = tensor_type(self.child, *args, **kwargs)  # type: ignore

    def replace_abstraction_top(
        self, tensor_type: type, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> None:
        self.child = tensor_type(*args, **kwargs)
