# stdlib
from functools import wraps


class AND:
    def __init__(self, op1, op2):
        self.op1 = op1
        self.op2 = op2

    def has_permission(self, node):
        return self.op1.has_permission(node) and self.op2.has_permission(node)


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


class OperationHolderMixin:
    def __and__(self, other):
        return OperandHolder(AND, self, other)

    def __or__(self, other):
        return OperandHolder(OR, self, other)

    def __rand__(self, other):
        return OperandHolder(AND, other, self)

    def __ror__(self, other):
        return OperandHolder(OR, other, self)

    def __invert__(self):
        return SingleOperandHolder(NOT, self)


class SingleOperandHolder(OperationHolderMixin):
    def __init__(self, operator_class, op1_class):
        self.operator_class = operator_class
        self.op1_class = op1_class

    def __call__(self, *args, **kwargs):
        op1 = self.op1_class(*args, **kwargs)
        return self.operator_class(op1)


class OperandHolder(OperationHolderMixin):
    def __init__(self, operator_class, op1_class, op2_class):
        self.operator_class = operator_class
        self.op1_class = op1_class
        self.op2_class = op2_class

    def __call__(self, *args, **kwargs):
        op1 = self.op1_class(*args, **kwargs)
        op2 = self.op2_class(*args, **kwargs)
        return self.operator_class(op1, op2)


class BasePermissionMetaclass(OperationHolderMixin, type):
    pass


class BasePermission(metaclass=BasePermissionMetaclass):
    """
    A base class from which all permission classes should inherit.
    """

    def has_permission(self, node, verify_key):
        """
        Return `True` if permission is granted, `False` otherwise.
        """
        return True


def check_permissions(permission_classes=[]):
    def decorator(func):
        @wraps(func)
        def wrapper_func(*args, **kwargs):
            for permission in permission_classes:
                if not permission.has_permission(*args, **kwargs):
                    raise Exception()
            return func(*args, **kwargs)

        return wrapper_func

    return decorator
