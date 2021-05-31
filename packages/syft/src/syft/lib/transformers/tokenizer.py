# syft relative
from ...proto.lib.python.string_pb2 import String as String_PB
from ...generate_wrapper import GenerateWrapper

# Third party
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer


def object2proto(obj: PreTrainedTokenizerFast) -> String_PB:
    # TODO Tokenizer.to_str serializes to a json string.
    # Protobuf has protobuf.json_format, which might be better than sending a raw string.
    protobuf_tokenizer = String_PB(data=obj._tokenizer.to_str())
    return protobuf_tokenizer


def proto2object(protobuf_tokenizer: String_PB) -> PreTrainedTokenizerFast:
    _tokenizer = Tokenizer.from_str(protobuf_tokenizer.data)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=_tokenizer)
    # TODO transformers tokenizer does not save special tokens in _tokenizer.to_str().
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer


GenerateWrapper(
    wrapped_type=PreTrainedTokenizerFast,
    import_path="transformers.PreTrainedTokenizerFast",
    protobuf_scheme=String_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
