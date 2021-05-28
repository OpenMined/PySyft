# syft relative
from ...proto.lib.python.string_pb2 import String as String_PB
from ...generate_wrapper import GenerateWrapper

# Third party
from tokenizers import Tokenizer


def object2proto(obj: object) -> String_PB:
    # TODO Tokenizer.to_str serializes to a json string.
    # Protobuf has protobuf.json_format, which might be better than sending a raw string.
    protobuf_tokenizer = String_PB(data=obj.to_str())
    return protobuf_tokenizer


def proto2object(protobuf_tokenizer: String_PB) -> Tokenizer:
    tokenizer = Tokenizer.from_str(protobuf_tokenizer.data)
    return tokenizer


GenerateWrapper(
    wrapped_type=Tokenizer,
    import_path="tokenizers.Tokenizer",
    protobuf_scheme=String_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
