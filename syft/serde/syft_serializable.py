def get_protobuf_subclasses(cls):
    """
        Function to retrieve all classes that implement the protobuf methods from the SyftSerializable class:

        A type that wants to implement the protobuf interface and to use it with syft-proto has
        to inherit SyftSerializable (directly or from the parent class) and to implement
        (cannot inherit from parent class):
            1. bufferize
            2. unbufferize
            3. get_protobuf_schema

        If these methods are not implemented, the class won't be enrolled in the types that can use syft-proto.
    """

    def check_implementation(s):
        bufferize_implemented = s.bufferize.__qualname__.startswith(s.__name__)
        unbufferize_implemented = s.unbufferize.__qualname__.startswith(s.__name__)
        get_protobuf_schema_implemented = s.get_protobuf_schema.__qualname__.startswith(s.__name__)
        return bufferize_implemented and unbufferize_implemented and get_protobuf_schema_implemented

    original_subclasses = {s for s in cls.__subclasses__() if check_implementation(s)}
    sub_sets = {
        s
        for c in cls.__subclasses__()
        for s in get_protobuf_subclasses(c)
        if check_implementation(s)
    }
    return original_subclasses.union(sub_sets)


def get_msgpack_subclasses(cls):
    """
        Function to retrieve all classes that implement the msgpack methods from the SyftSerializable class:

        A type that wants to implement the msgpack interface and to use it in syft has
        to inherit SyftSerializable (directly or from the parent class) and to implement
        (cannot inherit from parent class):
            1. simplify
            2. detail

        If these methods are not implemented, the class won't be enrolled in the types that can use msgpack.
    """

    def check_implementation(s):
        bufferize_implemented = s.simplify.__qualname__.startswith(s.__name__)
        unbufferize_implemented = s.detail.__qualname__.startswith(s.__name__)
        return bufferize_implemented and unbufferize_implemented

    original_subclasses = {s for s in cls.__subclasses__() if check_implementation(s)}
    sub_sets = {
        s
        for c in cls.__subclasses__()
        for s in get_msgpack_subclasses(c)
        if check_implementation(s)
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

        Note: the interface can be inherited from parent class, but each class has to write it's own explicit methods,
        even if they are the ones from the parent class.
    """

    @staticmethod
    def simplify(worker, obj):
        raise NotImplementedError

    @staticmethod
    def detail(worker, obj):
        raise NotImplementedError

    @staticmethod
    def bufferize(worker, obj):
        raise NotImplementedError

    @staticmethod
    def unbufferize(worker, obj):
        raise NotImplementedError

    @staticmethod
    def get_protobuf_schema():
        raise NotImplementedError
