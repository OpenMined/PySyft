# third party
from tokenizers import Tokenizer

# Third party
from transformers import PreTrainedTokenizerFast

# syft relative
from ... import deserialize
from ...generate_wrapper import GenerateWrapper
from ...lib.python.primitive_factory import PrimitiveFactory
from ...lib.python.util import upcast
from ...proto.lib.transformers.tokenizerfast_pb2 import (
    TokenizerFast as TokenizerFast_PB,
)


def object2proto(obj: PreTrainedTokenizerFast) -> TokenizerFast_PB:
    tokenizer_str = obj._tokenizer.to_str()
    tokenizer_str = PrimitiveFactory.generate_primitive(value=tokenizer_str)

    kwargs = obj.special_tokens_map
    kwargs["name_or_path"] = obj.name_or_path
    kwargs["padding_side"] = obj.padding_side
    kwargs["model_max_length"] = obj.model_max_length
    kwargs = PrimitiveFactory.generate_primitive(value=kwargs)

    protobuf_tokenizer = TokenizerFast_PB(
        id=tokenizer_str.id._object2proto(),
        tokenizer=tokenizer_str._object2proto(),
        kwargs=kwargs._object2proto(),
    )
    return protobuf_tokenizer


def proto2object(proto: TokenizerFast_PB) -> PreTrainedTokenizerFast:
    _tokenizer = Tokenizer.from_str(proto.tokenizer.data)
    kwargs = deserialize(proto.kwargs)
    kwargs = upcast(kwargs)

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=_tokenizer, **kwargs)
    return tokenizer


GenerateWrapper(
    wrapped_type=PreTrainedTokenizerFast,
    import_path="transformers.PreTrainedTokenizerFast",
    protobuf_scheme=TokenizerFast_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
