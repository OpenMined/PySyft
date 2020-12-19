# stdlib
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from syfertext.tokenizers.default_tokenizer import DefaultTokenizer

# syft relative
from ...core.store.storeable_object import StorableObject
from ...proto.lib.syfertext.default_tokenizer_pb2 import DefaultTokenizer as DefaultTokenizer_PB
from ..common.uid import UID

class DefaultTokenizerWrapper(StorableObject):

    def __init__(self, value: object):

        super().__init__(
            data = value,
            id = getattr(value, 'uuid'),
            tags = getattr(value, 'tags', {}),
            description = getattr(value, 'description', '')
        )

        self.value = value

    def _data_object2proto(self) -> DefaultTokenizer_PB:

        default_tokenizer_pb = DefaultTokenizer_PB()

        default_tokenizer.uuid = self.value.uuid.bytes
        default_tokenizer.prefixes.extend(self.value.prefixes)
        default_tokenizer.suffixes.extend(self.value.suffixes)
        default_tokenizer.infixes.extend(self.value.infixes)        



    @staticmethod
    def _data_proto2object(proto: DefaultTokenizer_PB) -> DefaultTokenizer:
        uuid = UUID(bytes=proto.uuid)
        prefixes = proto.prefixes
        suffixes = proto.suffixes
        infixes = proto.infixes

        default_tokenizer = DefaultTokenizer(uuid = uuid,
                                             prefixes = prefixes,
                                             suffixes = suffixes,
                                             infixes = infixes)

        


    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return DefaultTokenizer_PB
    
    @staticmethod
    def get_wrapped_type() -> type:
        return DefaultTokenizer
