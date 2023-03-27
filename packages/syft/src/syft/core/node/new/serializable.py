# stdlib
from typing import Any
from typing import Optional
from typing import Sequence

# syft absolute
import syft

# relative
from .recursive import recursive_serde_register

module_type = type(syft)


def serializable(
    attrs: Sequence[str] = [],
    without: Sequence[str] = [],
    inherit: Optional[bool] = True,
    inheritable: Optional[bool] = True,
    pydantic: Optional[bool] = False,
    **kwargs,
) -> Any:
    """
    Recursively serialize attributes of the class.

    Args:
        `attrs`       : List of attributes to serialize
        `without`     : List of attributes to exlucde from serialization
        `inherit`     : Whether to inherit serializable attribute list from base class
        `inheritable` : Whether the serializable attribute list can be inherited by derived class

    For non-pydantic classes,
        - `inheritable=True`  => Derived classes will include base class `attrs`
        - `inheritable=False` => Derived classes will not include base class `attrs`
        - `inherit=True`      => Base class `attrs` + `attrs` - `without`
        - `inherit=False`     => `attrs` - `without`

    For pydantic classes,
        - No need to provide `attrs`. They will be automatically inferred.
        - Providing `attrs` will override the inferred attributes.
        - `without` will work only on attributes of `Optional` type
        - `inherit`, `inheritable` will not work as pydantic inherits by default

    Returns:
        Decorated class
    """

    def rs_decorator(cls: Any) -> Any:
        recursive_serde_register(
            cls,
            serialize_attrs=attrs,
            exclude_attrs=without,
            inherit_attrs=inherit,
            inheritable_attrs=inheritable,
        )
        return cls

    return rs_decorator
