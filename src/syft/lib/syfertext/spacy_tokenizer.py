# stdlib
from typing import List
from typing import Optional
from uuid import UUID

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from syfertext.tokenizers import SpacyTokenizer

# syft relative
from ...core.common import UID
from ...core.store.storeable_object import StorableObject
from ...proto.lib.syfertext.spacy_tokenizer_pb2 import SpacyTokenizer as SpacyTokenizer_PB
from ...proto.lib.syfertext.spacy_tokenizer_pb2 import TokenSplits as TokenSplits_PB
from ...util import aggressive_set_attr


class SpacyTokenizerWrapper(StorableObject):

    def __init__(self, value: object):

        super().__init__(
            data = value,
            id = getattr(value, 'id', UID()),
            tags = getattr(value, 'tags', []),
            description = getattr(value, 'description', '')
        )

        self.value = value

    def _data_object2proto(self) -> SpacyTokenizer_PB:

        spacy_tokenizer_pb = SpacyTokenizer_PB()


        spacy_tokenizer_pb.uuid = self.id.value.bytes
        spacy_tokenizer_pb.prefixes.extend(self.value.prefixes)
        spacy_tokenizer_pb.suffixes.extend(self.value.suffixes)
        spacy_tokenizer_pb.infixes.extend(self.value.infixes)


        # Add the dictionary of exceptions to the ProtoBuf object
        for token, splits in self.value.exceptions.items():
            spacy_tokenizer_pb.exceptions[token].splits.extend(splits)        

            
        return spacy_tokenizer_pb

    @staticmethod
    def _data_proto2object(proto: SpacyTokenizer_PB) -> SpacyTokenizer:

        prefixes = proto.prefixes
        suffixes = proto.suffixes
        infixes = proto.infixes

        # Create a dict from the protobuf's map object representing the token
        # exceptions
        token_exceptions = dict()
        
        for token in proto.exceptions:
            token_splits = proto.exceptions[token].splits
            token_exceptions[token] = token_splits

        # Create the DefautTokenizer object
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

    
    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return SpacyTokenizer_PB
    
    @staticmethod
    def get_wrapped_type() -> type:
        return SpacyTokenizer

    @staticmethod
    def construct_new_object(
            id: UID,
            data: StorableObject,
            description: Optional[str],
            tags: Optional[List[str]],
    ) -> StorableObject:
        
        data.id = id
        data.tags = tags
        data.description = description
        
        return data
    
aggressive_set_attr(
    obj = SpacyTokenizer,
    name = 'serializable_wrapper_type',
    attr = SpacyTokenizerWrapper
)
