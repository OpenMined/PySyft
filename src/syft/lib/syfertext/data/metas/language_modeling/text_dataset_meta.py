# stdlib
from typing import List
from typing import Optional
from uuid import UUID

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from syfertext.data.metas.language_modeling import TextDatasetMeta

# syft relative
from ......core.common import UID
from ......generate_wrapper import GenerateWrapper
from ......core.store.storeable_object import StorableObject
from ......proto.lib.syfertext.data.metas.language_modeling.text_dataset_meta_pb2 import TextDatasetMeta as TextDatasetMeta_PB
from ......util import aggressive_set_attr


    
def object2proto(obj: TextDatasetMeta) -> TextDatasetMeta_PB:

    # For convenience, rename the object.
    text_dataset_meta = obj

    # Initialize the protobuf object for the tokenizer
    text_dataset_meta_pb = TextDatasetMeta_PB()

    text_dataset_meta_pb.uuid = text_dataset_meta.id.value.bytes
    text_dataset_meta_pb.train_path = text_dataset_meta.train_path
    text_dataset_meta_pb.valid_path = text_dataset_meta.valid_path
    text_dataset_meta_pb.test_path = text_dataset_meta.test_path        

    return text_dataset_meta_pb


def proto2object(proto: TextDatasetMeta_PB) -> TextDatasetMeta:

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



GenerateWrapper(
    wrapped_type=TextDatasetMeta,
    import_path="syfertext.data.metas.TextDatasetMeta",
    protobuf_scheme=TextDatasetMeta_PB,
    type_object2proto=object2proto,
    type_proto2object=proto2object,
)
