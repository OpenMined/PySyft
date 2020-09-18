import inspect
from typing import Callable


def get_from_inheritance_chain(cls: type, condition: Callable) -> set:
    """
    Generic function that extracts all nodes from the inheritance tree that respects
    a first order logic condition.
    """
    original_subclasses = {s for s in cls.__subclasses__() if condition(s)}
    sub_sets = {
        s
        for c in cls.__subclasses__()
        for s in get_from_inheritance_chain(c, condition)
        if condition(s)
    }
    return original_subclasses.union(sub_sets)


def get_protobuf_wrappers(cls: type) -> set:
    """
    Function to retrieve all wrappers that implement the protobuf methods from the
    SyftSerializable class:

    A type that wants to implement to wrap another type (eg. torch.Tensor) for the protobuf
    interface and to use it with syft-proto has to inherit SyftSerializable (directly or
    from the parent class) and to implement
    (cannot inherit from parent class):
        1. bufferize
        2. unbufferize
        3. get_protobuf_schema
        4. get_original_class
    If these methods are not implemented, the class won't be enrolled in the types that
    are wrappers can't use syft-proto.
    """

    def check_implementation(s):
        """
        Check if a class has:
            1. bufferize implemented.
            2. unbufferize implemented.
            3. get_protobuf_schema implemented.
            4. no abstact methods.
            5. get_original_class method
        To be sure that it can be used with protobufers.
        """
        not_abstract = not inspect.isabstract(s)
        bufferize_implemented = s.bufferize.__qualname__.startswith(s.__name__)
        unbufferize_implemented = s.unbufferize.__qualname__.startswith(s.__name__)
        get_protobuf_schema_implemented = s.get_protobuf_schema.__qualname__.startswith(s.__name__)
        get_original_class = s.get_original_class.__qualname__.startswith(s.__name__)
        return (
            not_abstract
            and bufferize_implemented
            and unbufferize_implemented
            and get_protobuf_schema_implemented
            and get_original_class
        )

    return get_from_inheritance_chain(cls, check_implementation)


def get_protobuf_classes(cls: type) -> set:
    """
    Function to retrieve all classes that implement the protobuf methods from the
    SyftSerializable class:

    A type that wants to implement the protobuf interface and to use it with syft-proto has
    to inherit SyftSerializable (directly or from the parent class) and to implement
    (cannot inherit from parent class):
        1. bufferize
        2. unbufferize
        3. get_protobuf_schema

    If these methods are not implemented, the class won't be enrolled in the types that can
    use syft-proto.
    """

    def check_implementation(s):
        """
        Check if a class has:
            1. bufferize implemented.
            2. unbufferize implemented.
            3. get_protobuf_schema implemented.
            4. no abstact methods.
            5. no get_original_class methods
        To be sure that it can be used with protobufers.
        """
        not_abstract = not inspect.isabstract(s)
        bufferize_implemented = s.bufferize.__qualname__.startswith(s.__name__)
        unbufferize_implemented = s.unbufferize.__qualname__.startswith(s.__name__)
        get_protobuf_schema_implemented = s.get_protobuf_schema.__qualname__.startswith(s.__name__)
        get_original_class = not s.get_original_class.__qualname__.startswith(s.__name__)
        return (
            not_abstract
            and bufferize_implemented
            and unbufferize_implemented
            and get_protobuf_schema_implemented
            and get_original_class
        )

    return get_from_inheritance_chain(cls, check_implementation)


def get_msgpack_subclasses(cls):
    """
    Function to retrieve all classes that implement the msgpack methods from the
    SyftSerializable class:

    A type that wants to implement the msgpack interface and to use it in syft has
    to inherit SyftSerializable (directly or from the parent class) and to implement
    (cannot inherit from parent class):
        1. simplify
        2. detail

    If these methods are not implemented, the class won't be enrolled in the types that
    can use msgpack.
    """

    def check_implementation(s):
        """
        Check if a class has:
            1. serialize implemented.
            2. detail implemented.

        To be sure that it can be used with msgpack.
        """
        not_abstract = not inspect.isabstract(s)
        bufferize_implemented = s.simplify.__qualname__.startswith(s.__name__)
        unbufferize_implemented = s.detail.__qualname__.startswith(s.__name__)
        return not_abstract and bufferize_implemented and unbufferize_implemented

    original_subclasses = {s for s in cls.__subclasses__() if check_implementation(s)}
    sub_sets = {
        s
        for c in cls.__subclasses__()
        for s in get_msgpack_subclasses(c)
        if check_implementation(s) and not inspect.isabstract(s)
    }
    return original_subclasses.union(sub_sets)


class SyftSerializable:
    """
    Interface for the communication protocols in syft.

    syft-proto methods:
        1. bufferize
        2. unbufferize
        3. get_protobuf_schema

    msgpack methods:
        1. simplify
        2. detail

    Note: the interface can be inherited from parent class, but each class
    has to write it's own explicit methods, even if they are the ones from the parent class.
    """

    @staticmethod
    def simplify(worker, obj):
        """
        Serialization method for msgpack.

        Parameters:
            worker: the worker on which the serialization is being made.
            obj: the object to be serialized, an instantiated type of
            the class that implements SyftSerializable.

        Returns:
            Serialized object using msgpack.
        """
        raise NotImplementedError

    @staticmethod
    def detail(worker, obj):
        """
        Deserialization method for msgpack.

        Parameters:
            worker: the worker on which the serialization is being made.
            obj: the object to be deserialized, a serialized object of
            the class that implements SyftSerializable.

        Returns:
            Serialized object using msgpack.
        """
        raise NotImplementedError

    @staticmethod
    def bufferize(worker, obj):
        """
        Serialization method for protobuf.

        Parameters:
            worker: the worker on which the bufferize is being made.
            obj: the object to be bufferized using protobufers, an instantiated type
            of the class that implements SyftSerializable.

        Returns:
            Protobuf class for the current type.
        """

        raise NotImplementedError

    @staticmethod
    def get_msgpack_code():
        """
        Method that provides a code for msgpack if the type is not present in proto.json.

        The returned object should be similar to:
        {
            "code": int value,
            "forced_code": int value
        }

        Both keys are optional, the common and right way would be to add only the "code" key.

        Returns:
            dict: A dict with the "code" or "forced_code" keys.
        """
        raise NotImplementedError

    @staticmethod
    def unbufferize(worker, obj):
        """
        Deserialization method for protobuf.

        Parameters:
            worker: the worker on which the unbufferize is being made.
            obj: the object to be unbufferized using protobufers, an instantiated type
            of the class that implements SyftSerializable.

        Returns:
            Protobuf class for the current type.
        """
        raise NotImplementedError

    @staticmethod
    def get_protobuf_schema():
        """
        Returns the protobuf schema used for this type.

        Returns:
            Protobuf type.
        """
        raise NotImplementedError

    @staticmethod
    def get_original_class():
        """
        Returns the original type, only used in wrappers.

        Returns:
            Wrapped type.
        """
        return NotImplementedError
