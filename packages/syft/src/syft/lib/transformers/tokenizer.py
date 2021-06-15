# third party
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

# syft relative
from ... import deserialize
from ...generate_wrapper import GenerateWrapper
from ...lib.python.primitive_factory import PrimitiveFactory
from ...lib.python.util import upcast
from ...proto.lib.transformers.tokenizerfast_pb2 import (
    TokenizerFast as TokenizerFast_PB,
)
from ..util import full_name_with_qualname


def object2proto(obj: PreTrainedTokenizerFast) -> TokenizerFast_PB:
    tokenizer_type = full_name_with_qualname(klass=type(obj))
    tokenizer_str = obj._tokenizer.to_str()

    kwargs = obj.special_tokens_map
    kwargs["name_or_path"] = obj.name_or_path
    kwargs["padding_side"] = obj.padding_side
    kwargs["model_max_length"] = obj.model_max_length
    kwargs = PrimitiveFactory.generate_primitive(value=kwargs)

    protobuf_tokenizer = TokenizerFast_PB(
        id=kwargs.id._object2proto(),
        tokenizer_type=tokenizer_type,
        tokenizer=tokenizer_str,
        kwargs=kwargs._object2proto(),
    )
    return protobuf_tokenizer


def proto2object(proto: TokenizerFast_PB) -> PreTrainedTokenizerFast:
    # TODO some subclasses of pretrainedtokenizerfast have required args,
    # cast every tokenizer as baseclass pretrainedtokenizerfast for now.
    # See  BertTokenizerFast.

    tokenizer_type = PreTrainedTokenizerFast

    _tokenizer = Tokenizer.from_str(proto.tokenizer)
    kwargs = deserialize(proto.kwargs)
    kwargs = upcast(kwargs)

    tokenizer = tokenizer_type(tokenizer_object=_tokenizer, **kwargs)
    return tokenizer


GenerateWrapper(
    wrapped_type=PreTrainedTokenizerFast,
    import_path="transformers.PreTrainedTokenizerFast",
    protobuf_scheme=TokenizerFast_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
