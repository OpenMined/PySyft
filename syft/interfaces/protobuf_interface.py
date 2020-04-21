def get_all_subclasses(cls):
    def check_implementation(s):
        bufferize_implemented = s.bufferize.__qualname__.startswith(s.__name__)
        unbufferize_implemented = s.unbufferize.__qualname__.startswith(s.__name__)
        get_protobuf_schema_implemented = s.get_protobuf_schema.__qualname__.startswith(s.__name__)
        return bufferize_implemented and unbufferize_implemented and get_protobuf_schema_implemented

    original_subclasses = {s for s in cls.__subclasses__() if check_implementation(s)}
    sub_sets = {s for c in cls.__subclasses__() for s in get_all_subclasses(c) if check_implementation(s)}
    return original_subclasses.union(sub_sets)

class ProtobufInterface:
    @staticmethod
    def bufferize():
        pass

    @staticmethod
    def unbufferize():
        pass

    @staticmethod
    def get_protobuf_schema():
        pass