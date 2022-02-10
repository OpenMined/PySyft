# stdlib
from typing import Optional

# third party
from nacl.signing import VerifyKey

# relative
from ...abstract.node_service_interface import NodeServiceInterface
from ..node_service.generic_payload.syft_message import NewSyftMessage


class AND:
    """Implements the `AND` functionality on a set of permission classes."""

    def __init__(self, op1, op2):
        self.op1 = op1
        self.op2 = op2

    def has_permission(self, *args, **kwargs) -> bool:
        return self.op1.has_permission(*args, **kwargs) and self.op2.has_permission(
            *args, **kwargs
        )

    def __call__(self):
        self.op1 = self.op1()
        self.op2 = self.op2()
        return self


class OR:
    """Implements the `OR` functionality on a set of permission classes."""

    def __init__(self, op1, op2):
        self.op1 = op1
        self.op2 = op2

    def has_permission(self, *args, **kwargs):
        return self.op1.has_permission(*args, **kwargs) or self.op2.has_permission(
            *args, **kwargs
        )

    def __call__(self):
        self.op1 = self.op1()
        self.op2 = self.op2()
        return self


class NOT:
    """Implements the `NOT` functionality on a permission class."""

    def __init__(self, op1):
        self.op1 = op1

    def has_permission(self, *args, **kwargs) -> bool:
        return not self.op1.has_permission(*args, **kwargs)

    def __call__(self):
        self.op1 = self.op1()
        return self


class BasePermissionMetaclass(type):
    """A metaclass to allow composition between different permission classes."""

    def __and__(self, other) -> AND:
        return AND(self, other)

    def __or__(self, other) -> OR:
        return OR(self, other)

    def __rand__(self, other) -> AND:
        return AND(other, self)

    def __ror__(self, other) -> OR:
        return OR(other, self)

    def __invert__(self) -> NOT:
        return NOT(self)


class BasePermission(metaclass=BasePermissionMetaclass):
    """A base class from which all permission classes should inherit."""

    def has_permission(
        self,
        msg: NewSyftMessage,
        node: NodeServiceInterface,
        verify_key: Optional[VerifyKey],
    ) -> bool:
        """Return `True` if permission is granted, `False` otherwise."""
        raise NotImplementedError
