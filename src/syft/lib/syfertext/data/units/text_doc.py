# stdlib
from typing import List
from typing import Optional
from uuid import UUID

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from syfertext.data.units import TextDoc
from syfertext.data.units import TokenMeta

# syft relative
from .....core.common import UID
from .....core.store.storeable_object import StorableObject
from .....proto.lib.syfertext.data.units.text_doc_pb2 import TextDoc as TextDoc_PB
from .....proto.lib.syfertext.data.units.token_meta_pb2 import TokenMeta as TokenMeta_PB
from .....proto.lib.syfertext.data.units.attribute_values_pb2 import AttributeValues as AttributeValues_PB
from .....util import aggressive_set_attr


class TextDocWrapper(StorableObject):

    def __init__(self, value: object):

        super().__init__(
            data = value,
            id = getattr(value, 'id', UID()),
            tags = getattr(value, 'tags', []),
            description = getattr(value, 'description', '')
        )

        self.text_doc = value

    def _data_object2proto(self) -> TokenMeta_PB:

        # Initialize the TextDoc protobuf object
        text_doc_pb = TextDoc_PB()

        # Set an ID of the text doc object
        text_doc_pb.uuid = self.id.value.bytes
        
        # Prepare the token meta protobuf objects to be added
        # to the list of token metas of the text doc
        # protobuf
        for token_meta in self.text_doc.token_metas:

            # Create a TokenMeta protobuf object
            token_meta_pb = TokenMeta_PB()

            token_meta_pb.text = token_meta.text
            token_meta_pb.space_after = token_meta.space_after

            # Add the dictionary of token meta attributes
            # to the token meta protobuf object
            for att_name, att_values in token_meta.attributes.items():
                token_meta_pb.attributes[att_name].values.extend(att_values)

            # Append the token meta protobuf object to the list
            # of token metas in the text doc protobuf object
            text_doc_pb.token_metas.append(token_meta_pb)

        # Now start filling the attributes of the text doc
        # protobuf object
        for att_name, att_values in self.text_doc.attributes.items():
            text_doc_pb.attributes[att_name].values.extend(att_values)
        
        return text_doc_pb

    @staticmethod
    def _data_proto2object(proto: TextDoc_PB) -> TextDoc:

        # For convenience, rename the `proto` argument
        text_doc_pb = proto
        
        # Initialize the TextDoc object
        text_doc = TextDoc()

        ## Load the TokenMeta objects into the TextDoc's
        ## token_metas attribute
        
        for token_meta_pb in text_doc_pb.token_metas:

            text = token_meta_pb.text
            space_after = token_meta_pb.space_after
            
            # Initialize the attributes dict
            attributes = dict()

            # Iterate over the keys of the attributes protobuf object
            for att_name in token_meta_pb.attributes:
                att_values = token_meta_pb.attributes[att_name].values
                attributes[att_name] = att_values

            # Now create the TokenMeta object
            token_meta = TokenMeta(text = text, space_after = space_after)
            token_meta.attributes = attributes

            # Add the TokenMeta object to the TextDoc object
            text_doc.token_metas.append(token_meta)


        ## Load the TextDoc attributes property
        
        # Initialize the attributes dict
        attributes = dict()

        # Iterate over the keys of the attributes protobuf object
        for att_name in text_doc_pb.attributes:
            att_values = text_doc_pb.attributes[att_name].values
            text_doc.attributes[att_name] = att_values

        
        # Create a uuid.UUID object and assign it as an attribute to the TextDoc
        # The reason I do not set the id directly in the constructor is because I
        # do not want the API to expose the ID which is not something the end user
        # should worry about.
        uuid = UUID(bytes = text_doc_pb.uuid)
        text_doc.id = UID(value = uuid)

        return text_doc

    
    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return TextDoc_PB
    
    @staticmethod
    def get_wrapped_type() -> type:
        return TextDoc

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
    obj = TextDoc,
    name = 'serializable_wrapper_type',
    attr = TextDocWrapper
)
