# stdlib
from typing import List
from typing import Optional
from uuid import UUID

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from syfertext.data.metas.language_modeling import TextDatasetMeta

# syft relative
from ......core.common import UID
from ......core.store.storeable_object import StorableObject
from ......proto.lib.syfertext.data.metas.language_modeling.text_dataset_meta_pb2 import TextDatasetMeta as TextDatasetMeta_PB
from ......util import aggressive_set_attr


class TextDatasetMetaWrapper(StorableObject):

    def __init__(self, value: object):

        super().__init__(
            data = value,
            id = getattr(value, 'id', UID()),
            tags = getattr(value, 'tags', []),
            description = getattr(value, 'description', '')
        )

        self.text_dataset_meta = value

    
    def _data_object2proto(self) -> TextDatasetMeta_PB:

        # Initialize the protobuf object for the tokenizer
        text_dataset_meta_pb = TextDatasetMeta_PB()

        text_dataset_meta_pb.uuid = self.id.value.bytes        
        text_dataset_meta_pb.train_path = self.text_dataset_meta.train_path
        text_dataset_meta_pb.valid_path = self.text_dataset_meta.valid_path
        text_dataset_meta_pb.test_path = self.text_dataset_meta.test_path        
            
        return text_dataset_meta_pb
    
    @staticmethod
    def _data_proto2object(proto: TextDatasetMeta_PB) -> TextDatasetMeta:

        # Renaming the variable just for convenience
        text_dataset_meta_pb = proto

        
        # Create the TextDatasetMeta object
        text_dataset_meta = TextDatasetMeta(train_path = text_dataset_meta_pb.train_path,
                                            valid_path = text_dataset_meta_pb.valid_path,
                                            test_path = text_dataset_meta_pb.test_path,
                                         )
        

        # Create a uid
        uuid = UUID(bytes = text_dataset_meta_pb.uuid)
        text_dataset_meta.id = UID(value = uuid)

        return text_dataset_meta

    
    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return TextDatasetMeta_PB
    
    @staticmethod
    def get_wrapped_type() -> type:
        return TextDatasetMeta

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
    obj = TextDatasetMeta,
    name = 'serializable_wrapper_type',
    attr = TextDatasetMetaWrapper
)
