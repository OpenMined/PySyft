# stdlib
from typing import Dict
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from typing_extensions import final

# relative
from ...... import serialize
from ......proto.grid.messages.dataset_messages_pb2 import (
    CreateDatasetMessage as CreateDatasetMessage_PB,
)
from ......proto.grid.messages.dataset_messages_pb2 import (
    DeleteDatasetMessage as DeleteDatasetMessage_PB,
)
from ......proto.grid.messages.dataset_messages_pb2 import (
    GetDatasetMessage as GetDatasetMessage_PB,
)
from ......proto.grid.messages.dataset_messages_pb2 import (
    GetDatasetResponse as GetDatasetResponse_PB,
)
from ......proto.grid.messages.dataset_messages_pb2 import (
    GetDatasetsMessage as GetDatasetsMessage_PB,
)
from ......proto.grid.messages.dataset_messages_pb2 import (
    GetDatasetsResponse as GetDatasetsResponse_PB,
)
from ......proto.grid.messages.dataset_messages_pb2 import (
    UpdateDatasetMessage as UpdateDatasetMessage_PB,
)
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.deserialize import _deserialize
from .....common.serde.serializable import serializable
from .....common.uid import UID
from .....io.address import Address


@final
@serializable()
class CreateDatasetMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        dataset: bytes,
        metadata: Dict[str, str],
        reply_to: Address,
        platform: str,
        msg_id: Optional[UID] = None,
    ) -> None:
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.dataset = dataset
        self.metadata = metadata
        self.platform = platform

    def _object2proto(self) -> CreateDatasetMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: CreateDatasetMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return CreateDatasetMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            dataset=self.dataset,
            metadata=self.metadata,
            reply_to=serialize(self.reply_to),
            platform=self.platform,
        )

    @staticmethod
    def _proto2object(
        proto: CreateDatasetMessage_PB,
    ) -> "CreateDatasetMessage":
        """Creates a CreateDatasetMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: CreateDatasetMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return CreateDatasetMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            dataset=proto.dataset,
            metadata=dict(proto.metadata),
            reply_to=_deserialize(blob=proto.reply_to),
            platform=proto.platform,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        """Return the type of protobuf object which stores a class of this type
        As a part of serialization and deserialization, we need the ability to
        lookup the protobuf object type directly from the object type. This
        static method allows us to do this.
        Importantly, this method is also used to create the reverse lookup ability within
        the metaclass of Serializable. In the metaclass, it calls this method and then
        it takes whatever type is returned from this method and adds an attribute to it
        with the type of this class attached to it. See the MetaSerializable class for
        details.
        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        return CreateDatasetMessage_PB


@final
@serializable()
class GetDatasetMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        dataset_id: str,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.dataset_id = dataset_id

    def _object2proto(self) -> GetDatasetMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: GetDatasetMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetDatasetMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            dataset_id=self.dataset_id,
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: GetDatasetMessage_PB,
    ) -> "GetDatasetMessage":
        """Creates a GetDatasetMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: GetDatasetMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetDatasetMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            dataset_id=proto.dataset_id,
            reply_to=_deserialize(blob=proto.reply_to),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        """Return the type of protobuf object which stores a class of this type
        As a part of serialization and deserialization, we need the ability to
        lookup the protobuf object type directly from the object type. This
        static method allows us to do this.
        Importantly, this method is also used to create the reverse lookup ability within
        the metaclass of Serializable. In the metaclass, it calls this method and then
        it takes whatever type is returned from this method and adds an attribute to it
        with the type of this class attached to it. See the MetaSerializable class for
        details.
        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        return GetDatasetMessage_PB


@final
@serializable()
class GetDatasetResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        metadata: Dict,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.metadata = metadata

    def _object2proto(self) -> GetDatasetResponse_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: SignalingOfferMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetDatasetResponse_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            metadata=self.metadata,
        )

    @staticmethod
    def _proto2object(
        proto: GetDatasetResponse_PB,
    ) -> "GetDatasetResponse":
        """Creates a SignalingOfferMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: SignalingOfferMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetDatasetResponse(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            metadata=dict(proto.metadata),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        """Return the type of protobuf object which stores a class of this type
        As a part of serialization and deserialization, we need the ability to
        lookup the protobuf object type directly from the object type. This
        static method allows us to do this.
        Importantly, this method is also used to create the reverse lookup ability within
        the metaclass of Serializable. In the metaclass, it calls this method and then
        it takes whatever type is returned from this method and adds an attribute to it
        with the type of this class attached to it. See the MetaSerializable class for
        details.
        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        return GetDatasetResponse_PB


@final
@serializable()
class GetDatasetsMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)

    def _object2proto(self) -> GetDatasetsMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: GetDatasetsMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetDatasetsMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: GetDatasetsMessage_PB,
    ) -> "GetDatasetsMessage":
        """Creates a GetDatasetsMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: GetDatasetsMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetDatasetsMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            reply_to=_deserialize(blob=proto.reply_to),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        """Return the type of protobuf object which stores a class of this type
        As a part of serialization and deserialization, we need the ability to
        lookup the protobuf object type directly from the object type. This
        static method allows us to do this.
        Importantly, this method is also used to create the reverse lookup ability within
        the metaclass of Serializable. In the metaclass, it calls this method and then
        it takes whatever type is returned from this method and adds an attribute to it
        with the type of this class attached to it. See the MetaSerializable class for
        details.
        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        return GetDatasetsMessage_PB


@final
@serializable()
class GetDatasetsResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        metadatas: List[Dict],
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.metadatas = metadatas

    def _object2proto(self) -> GetDatasetsResponse_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: SignalingOfferMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """

        msg = GetDatasetsResponse_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
        )

        metadata_container = GetDatasetsResponse_PB.metadata_container

        for _metadata in self.metadatas:
            metadata = {}
            for k, v in _metadata.items():
                if not isinstance(v, bytes):
                    metadata[k] = serialize(v, to_bytes=True)
                else:
                    metadata[k] = v

            cm = metadata_container(metadata=metadata)
            msg.metadatas.append(cm)

        return msg

    @staticmethod
    def _proto2object(
        proto: GetDatasetsResponse_PB,
    ) -> "GetDatasetsResponse":
        """Creates a SignalingOfferMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: SignalingOfferMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetDatasetsResponse(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            metadatas=[
                dict(metadata_container.metadata)
                for metadata_container in proto.metadatas
            ],
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        """Return the type of protobuf object which stores a class of this type
        As a part of serialization and deserialization, we need the ability to
        lookup the protobuf object type directly from the object type. This
        static method allows us to do this.
        Importantly, this method is also used to create the reverse lookup ability within
        the metaclass of Serializable. In the metaclass, it calls this method and then
        it takes whatever type is returned from this method and adds an attribute to it
        with the type of this class attached to it. See the MetaSerializable class for
        details.
        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        return GetDatasetsResponse_PB


@final
@serializable()
class UpdateDatasetMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        dataset_id: str,
        metadata: Dict,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.metadata = metadata
        self.dataset_id = dataset_id

    def _object2proto(self) -> UpdateDatasetMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: UpdateDatasetMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return UpdateDatasetMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            dataset_id=self.dataset_id,
            metadata=self.metadata,
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: UpdateDatasetMessage_PB,
    ) -> "UpdateDatasetMessage":
        """Creates a UpdateDatasetMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: UpdateDatasetMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return UpdateDatasetMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            dataset_id=proto.dataset_id,
            metadata=dict(proto.metadata),
            reply_to=_deserialize(blob=proto.reply_to),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        """Return the type of protobuf object which stores a class of this type
        As a part of serialization and deserialization, we need the ability to
        lookup the protobuf object type directly from the object type. This
        static method allows us to do this.
        Importantly, this method is also used to create the reverse lookup ability within
        the metaclass of Serializable. In the metaclass, it calls this method and then
        it takes whatever type is returned from this method and adds an attribute to it
        with the type of this class attached to it. See the MetaSerializable class for
        details.
        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        return UpdateDatasetMessage_PB


@final
@serializable()
class DeleteDatasetMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        dataset_id: str,
        reply_to: Address,
        bin_object_id: Optional[str] = None,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.dataset_id = dataset_id
        self.bin_object_id = bin_object_id

    def _object2proto(self) -> DeleteDatasetMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: DeleteDatasetMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return DeleteDatasetMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            dataset_id=self.dataset_id,
            bin_object_id=self.bin_object_id,
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: DeleteDatasetMessage_PB,
    ) -> "DeleteDatasetMessage":
        """Creates a DeleteDatasetMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: DeleteDatasetMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return DeleteDatasetMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            dataset_id=proto.dataset_id,
            bin_object_id=proto.bin_object_id,
            reply_to=_deserialize(blob=proto.reply_to),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        """Return the type of protobuf object which stores a class of this type
        As a part of serialization and deserialization, we need the ability to
        lookup the protobuf object type directly from the object type. This
        static method allows us to do this.
        Importantly, this method is also used to create the reverse lookup ability within
        the metaclass of Serializable. In the metaclass, it calls this method and then
        it takes whatever type is returned from this method and adds an attribute to it
        with the type of this class attached to it. See the MetaSerializable class for
        details.
        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType
        """

        return DeleteDatasetMessage_PB
