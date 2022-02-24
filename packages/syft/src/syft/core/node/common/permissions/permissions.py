# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Optional
from typing import Type

# third party
from nacl.signing import VerifyKey

# relative
from ...abstract.node_service_interface import NodeServiceInterface
from ..node_service.generic_payload.syft_message import NewSyftMessage as SyftMessage


class BinaryOperation:
    """Executes an operation between two operands."""

    def __init__(self, op1: Any, op2: Any, operator: Any) -> None:
        self.op1 = op1
        self.op2 = op2
        self.operator = operator

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        op1 = self.op1(*args, **kwargs)
        op2 = self.op2(*args, **kwargs)
        return self.operator(op1, op2)


class UnaryOperation:
    """Executes an operation on a single operand."""

    def __init__(self, op1: Any, operator: Any) -> None:
        self.op1 = op1
        self.operator = operator

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        op1 = self.op1(*args, **kwargs)
        return self.operator(op1)


class AND:
    """Implements the `AND` functionality on a set of permission classes."""

    def __init__(self, op1: Type[BasePermission], op2: Type[BasePermission]):
        self.op1 = op1
        self.op2 = op2

    def has_permission(self, *args: Any, **kwargs: Any) -> bool:
        return self.op1.has_permission(*args, **kwargs) and self.op2.has_permission(
            *args, **kwargs
        )


class OR:
    """Implements the `OR` functionality on a set of permission classes."""

    def __init__(self, op1: Type[BasePermission], op2: Type[BasePermission]):
        self.op1 = op1
        self.op2 = op2

    def has_permission(self, *args: Any, **kwargs: Any) -> bool:
        return self.op1.has_permission(*args, **kwargs) or self.op2.has_permission(
            *args, **kwargs
        )


class NOT:
    """Implements the `NOT` functionality on a permission class."""

    def __init__(self, op1: Type[BasePermission]):
        self.op1 = op1

    def has_permission(self, *args: Any, **kwargs: Any) -> bool:
        return not self.op1.has_permission(*args, **kwargs)


class BasePermissionMetaclass(type):
    """A metaclass to allow composition between different permission classes."""

    def __and__(self, other: Type[BasePermission]) -> BinaryOperation:
        return BinaryOperation(self, other, AND)

    def __or__(self, other: Type[BasePermission]) -> BinaryOperation:  # type: ignore
        return BinaryOperation(self, other, OR)

    def __rand__(self, other: Type[BasePermission]) -> BinaryOperation:
        return BinaryOperation(other, self, AND)

    def __ror__(self, other: Type[BasePermission]) -> BinaryOperation:  # type: ignore
        return BinaryOperation(other, self, OR)

    def __invert__(self) -> UnaryOperation:
        return UnaryOperation(self, NOT)


class BasePermission(metaclass=BasePermissionMetaclass):
    """A base class from which all permission classes should inherit."""

    def has_permission(
        self,
        msg: SyftMessage,
        node: NodeServiceInterface,
        verify_key: Optional[VerifyKey],
    ) -> bool:
        """Return `True` if permission is granted, `False` otherwise."""
        raise NotImplementedError
