# stdlib
from typing import Any
from typing import Sequence

# syft absolute
import syft

# relative
from .recursive import recursive_serde_register

module_type = type(syft)


def serializable(
    recursive_serde: bool = True,
    attrs: Sequence[str] = [],
    inherit_attrs: bool = True,
    **kwargs,
) -> Any:
    """ """

    def rs_decorator(cls: Any) -> Any:
        if recursive_serde:
            recursive_serde_register(
                cls, state_attrs=attrs, inherit_attrs=inherit_attrs
            )
        return cls

    return rs_decorator
