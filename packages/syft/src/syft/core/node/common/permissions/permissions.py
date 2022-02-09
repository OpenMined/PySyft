# stdlib
from functools import wraps
from typing import List


class AND:
    def __init__(self, op1, op2):
        self.op1 = op1
        self.op2 = op2

    def has_permission(self, node, verify_key):
        return self.op1.has_permission(node, verify_key) and self.op2.has_permission(
            node, verify_key
        )


class OR:
    def __init__(self, op1, op2):
        self.op1 = op1
        self.op2 = op2

    def has_permission(self, node, verify_key):
        return self.op1.has_permission(node, verify_key) or self.op2.has_permission(
            node, verify_key
        )


class NOT:
    def __init__(self, op1):
        self.op1 = op1

    def has_permission(self, node, verify_key):
        return not self.op1.has_permission(node, verify_key)


class BasePermissionMetaclass(type):
    def __and__(self, other):
        return AND(self, other)

    def __or__(self, other):
        return OR(self, other)

    def __rand__(self, other):
        return AND(other, self)

    def __ror__(self, other):
        return OR(other, self)

    def __invert__(self):
        return NOT(self)


class BasePermission(metaclass=BasePermissionMetaclass):
    """A base class from which all permission classes should inherit."""

    def has_permission(self, node, verify_key):
        """
        Return `True` if permission is granted, `False` otherwise.
        """
        return True


def check_permissions(permission_classes: List = []):
    """A decorator to check the given the permission classes are satisfied.
        Raises an appropriate exception if one of the permission is not permitted.

    Args:
        permission_classes (List, optional): List of permission classes. Defaults to [].
    """

    def decorator(func):
        @wraps(func)
        def wrapper_func(*args, **kwargs):
            for permission_class in permission_classes:
                if not permission_class.has_permission(*args, **kwargs):
                    raise Exception()
            return func(*args, **kwargs)

        return wrapper_func

    return decorator
