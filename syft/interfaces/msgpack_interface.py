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


class MsgpackInterface:
    @staticmethod
    def simplify(worker, obj):
        pass

    @staticmethod
    def detail(worker, obj):
        pass


print(MsgpackInterface.__subclasses__())
