# syft relative
from ...proto.lib.python.string_pb2 import String as String_PB
from ...generate_wrapper import GenerateWrapper

# Third party
from tokenizers import Encoding


def object2proto(obj: object) -> String_PB:
    # TODO __getstate__ and __setstate__ return bytes, might be better protobuf for that
    protobuf_obj = String_PB(data=obj.__getstate__().decode('utf-8'))
    return protobuf_obj


def proto2object(protobuf_encoding: String_PB) -> Encoding:
    encoding = Encoding()
    encoding.__setstate__(protobuf_encoding.data.encode('utf-8'))
    return encoding


GenerateWrapper(
    wrapped_type=Encoding,
    import_path="tokenizers.Encoding",
    protobuf_scheme=String_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
