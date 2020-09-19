import inspect
from typing import Callable

from .serializable import Serializable

def get_from_inheritance_chain(condition: Callable, cls: type = Serializable) -> set:
    """
        Generic function that extracts all nodes from the inheritance tree that respects
        a first order logic condition.
    """
    original_subclasses = {s for s in cls.__subclasses__() if condition(s)}
    sub_sets = {
        s
        for c in cls.__subclasses__()
        for s in get_from_inheritance_chain(condition, c)
        if condition(s)
    }
    return original_subclasses.union(sub_sets)

def check_implementation(s):
    """
        Check if a class has:
            1. bufferize implemented.
            2. unbufferize implemented.
            3. get_protobuf_schema implemented.
            4. no abstact methods.
            5. no get_wrapped_type method
        To be sure that it can be used with protobufers.
    """
    not_abstract = not inspect.isabstract(s)
    bufferize_implemented = hasattr(s, '_object2proto')
    unbufferize_implemented = hasattr(s, '_proto2object')
    is_wrapper_implemented = hasattr(s, 'is_wrapper')
    wrapped_class_implemented = hasattr(s, 'wrapped_class')
    return (
        not_abstract
        and bufferize_implemented
        and unbufferize_implemented
        and is_wrapper_implemented
        and wrapped_class_implemented
    )

def get_protobuf_wrappers(cls: type = Serializable) -> set:
    """
        Function to retrieve all wrappers that implement the protobuf methods from the
        SyftSerializable class:
        A type that wants to implement to wrap another type (eg. torch.Tensor) for the protobuf
        interface and to use it with syft-proto has to inherit SyftSerializable (directly or
        from the parent class) and to implement
        (cannot inherit from parent class):
            1. to_protobuf
            2. from_protobuf
            3. get_protobuf_schema
            4. get_wrapped_type
        If these methods are not implemented, the class won't be enrolled in the types that
        are wrappers and won't be able to use syft-proto.
    """

    return get_from_inheritance_chain(check_implementation, cls)


def get_protobuf_classes(cls: type = Serializable) -> set:
    """
        Function to retrieve all classes that implement the protobuf methods from the
        SyftSerializable class:
        A type that wants to implement the protobuf interface and to use it with syft-proto has
        to inherit SyftSerializable (directly or from the parent class) and to implement
        (cannot inherit from parent class):
            1. to_protobuf
            2. from_protobuf
            3. get_protobuf_schema
        If these methods are not implemented, the class won't be enrolled in the types that can
        use syft-proto.
    """

    return get_from_inheritance_chain(check_implementation, cls)
