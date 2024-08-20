# stdlib
from collections.abc import Callable
from typing import TypeVar

# syft absolute
import syft

# relative
from .recursive import recursive_serde_register

module_type = type(syft)

T = TypeVar("T", bound=type)


def serializable(
    attrs: list[str] | None = None,
    without: list[str] | None = None,
    inherit: bool | None = True,
    inheritable: bool | None = True,
    canonical_name: str | None = None,
    version: int | None = None,
) -> Callable[[T], T]:
    """
    Recursively serialize attributes of the class.

    Args:
        attrs (list[str] | None): List of attributes to serialize. Defaults to None.
        without (list[str] | None): List of attributes to exclude from serialization. Defaults to None.
        inherit (bool | None): Whether to inherit the serializable attribute list from the base class. Defaults to True.
        inheritable (bool | None): Whether the serializable attribute list can be inherited by derived
         classes. Defaults to True.
        canonical_name (str | None): The canonical name for the serialization. Defaults to None.
        version (int | None): The version number for the serialization. Defaults to None.

    For non-pydantic classes:
        - `inheritable=True`  => Derived classes will include base class `attrs`.
        - `inheritable=False` => Derived classes will not include base class `attrs`.
        - `inherit=True`      => Base class `attrs` + `attrs` - `without`.
        - `inherit=False`     => `attrs` - `without`.

    For pydantic classes:
        - No need to provide `attrs`. They will be automatically inferred.
        - Providing `attrs` will override the inferred attributes.
        - `without` will work only on attributes of `Optional` type.
        - `inherit`, `inheritable` will not work as pydantic inherits by default.

    Returns:
        Callable[[T], T]: The decorated class.
    """

    def rs_decorator(cls: T) -> T:
        recursive_serde_register(
            cls,
            serialize_attrs=attrs,
            exclude_attrs=without,
            inherit_attrs=inherit,
            inheritable_attrs=inheritable,
            canonical_name=canonical_name,
            version=version,
        )
        return cls

    return rs_decorator
