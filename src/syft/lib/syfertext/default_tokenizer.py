# stdlib
from typing import List
from typing import Optional
from uuid import UUID

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from syfertext.tokenizers import DefaultTokenizer

# syft relative
from ...core.common import UID
from ...core.store.storeable_object import StorableObject
from ...proto.lib.syfertext.default_tokenizer_pb2 import DefaultTokenizer as DefaultTokenizer_PB
from ...util import aggressive_set_attr


class DefaultTokenizerWrapper(StorableObject):

    def __init__(self, value: object):

        super().__init__(
            data = value,
            id = getattr(value, 'id', UID()),
            tags = getattr(value, 'tags', []),
            description = getattr(value, 'description', '')
        )

        self.value = value

    def _data_object2proto(self) -> DefaultTokenizer_PB:

        default_tokenizer_pb = DefaultTokenizer_PB()

        default_tokenizer_pb.uuid = self.id.value.bytes
        default_tokenizer_pb.prefixes.extend(self.value.prefixes)
        default_tokenizer_pb.suffixes.extend(self.value.suffixes)
        default_tokenizer_pb.infixes.extend(self.value.infixes)        


        return default_tokenizer_pb

    @staticmethod
    def _data_proto2object(proto: DefaultTokenizer_PB) -> DefaultTokenizer:

        prefixes = proto.prefixes
        suffixes = proto.suffixes
        infixes = proto.infixes

        default_tokenizer = DefaultTokenizer(prefixes = prefixes,
                                             suffixes = suffixes,
                                             infixes = infixes)
        
        # Create a uuid.UUID object and assign it as an attribute to the tokenizer
        # The reason I do not set the id directly in the constructor is because I
        # do not want the API to expose the ID which is not something the end user
        # should worry about.
        uuid = UUID(bytes = proto.uuid)
        default_tokenizer.id = UID(value = uuid)

        return default_tokenizer

    
    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return DefaultTokenizer_PB
    
    @staticmethod
    def get_wrapped_type() -> type:
        return DefaultTokenizer

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
    obj = DefaultTokenizer,
    name = 'serializable_wrapper_type',
    attr = DefaultTokenizerWrapper
)
