# stdlib
from typing import Any

# syft absolute
import syft

# relative
from .recursive import recursive_serde_register

module_type = type(syft)


def serializable(
    recursive_serde: bool = False,
) -> Any:
    def rs_decorator(cls: Any) -> Any:
        recursive_serde_register(cls)
        return cls

    if recursive_serde:
        return rs_decorator
