def get_protobuf_subclasses(cls):
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
