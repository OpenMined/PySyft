# stdlib
from typing import List
from typing import Optional
from uuid import UUID

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from syfertext.tokenizers import SpacyTokenizer

# syft relative
from ....core.common import UID
from ....core.store.storeable_object import StorableObject
from ....generate_wrapper import GenerateWrapper
from ....proto.lib.syfertext.tokenizers.spacy_tokenizer_pb2 import SpacyTokenizer as SpacyTokenizer_PB
from ....proto.lib.syfertext.tokenizers.spacy_tokenizer_pb2 import TokenSplits as TokenSplits_PB
from ....util import aggressive_set_attr


def object2proto(obj: SpacyTokenizer, obj_pb: SpacyTokenizer_PB = None) -> SpacyTokenizer_PB:
    """utility method to convert a SpacyTokenizer object into
    a protobuf boject.

    This is not required by syft. It is meant to be used by
    any external function that needs to convert a spacy tokenizer
    into a protobu object.
    """

    # For convenience, rename the object.
    tokenizer = obj

    if obj_pb is None:
        
        # Initialize the protobuf object for the tokenizer
        tokenizer_pb = SpacyTokenizer_PB()
    else:

        # Use the passed protobuf object
        tokenizer_pb = obj_pb        

    tokenizer_pb.uuid = getattr(tokenizer, 'id', UID()).value.bytes
    tokenizer_pb.prefixes.extend(tokenizer.prefixes)
    tokenizer_pb.suffixes.extend(tokenizer.suffixes)
    tokenizer_pb.infixes.extend(tokenizer.infixes)


    # Add the dictionary of exceptions to the ProtoBuf object
    for token, splits in tokenizer.exceptions.items():
        tokenizer_pb.exceptions[token].splits.extend(splits)        


    return tokenizer_pb

def proto2object(proto: SpacyTokenizer_PB) -> SpacyTokenizer:

    prefixes = proto.prefixes
    suffixes = proto.suffixes
    infixes = proto.infixes

    # Create a dict from the protobuf's map object representing the token
    # exceptions
    token_exceptions = dict()

    for token in proto.exceptions:
        token_splits = proto.exceptions[token].splits
        token_exceptions[token] = token_splits

    # Create the SpacyTokenizer object
    spacy_tokenizer = SpacyTokenizer(prefixes = prefixes,
                                     suffixes = suffixes,
                                     infixes = infixes,
                                     exceptions = token_exceptions
                                     )

    # Create a uuid.UUID object and assign it as an attribute to the tokenizer
    # The reason I do not set the id directly in the constructor is because I
    # do not want the API to expose the ID which is not something the end user
    # should worry about.
    uuid = UUID(bytes = proto.uuid)
    spacy_tokenizer.id = UID(value = uuid)

    return spacy_tokenizer


GenerateWrapper(
    wrapped_type=SpacyTokenizer,
    import_path="syfertext.tokenizers.SpacyTokenizer",
    protobuf_scheme=SpacyTokenizer_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
