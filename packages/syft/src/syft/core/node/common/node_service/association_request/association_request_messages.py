# stdlib
import json
from typing import Dict
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from typing_extensions import final

# syft absolute
from syft import serialize
from syft.core.common.message import ImmediateSyftMessageWithReply
from syft.core.common.message import ImmediateSyftMessageWithoutReply
from syft.core.common.serde.deserialize import _deserialize
from syft.core.common.uid import UID
from syft.core.io.address import Address
from syft.core.node.common.client import AbstractNodeClient
from syft.core.node.common.client import Client
from syft.lib.python import Dict as SyftDict
from syft.proto.grid.messages.association_messages_pb2 import (
    DeleteAssociationRequestMessage as DeleteAssociationRequestMessage_PB,
)
from syft.proto.grid.messages.association_messages_pb2 import (
    GetAssociationRequestMessage as GetAssociationRequestMessage_PB,
)
from syft.proto.grid.messages.association_messages_pb2 import (
    GetAssociationRequestResponse as GetAssociationRequestResponse_PB,
)
from syft.proto.grid.messages.association_messages_pb2 import (
    GetAssociationRequestsMessage as GetAssociationRequestsMessage_PB,
)
from syft.proto.grid.messages.association_messages_pb2 import (
    GetAssociationRequestsResponse as GetAssociationRequestsResponse_PB,
)
from syft.proto.grid.messages.association_messages_pb2 import (
    ReceiveAssociationRequestMessage as ReceiveAssociationRequestMessage_PB,
)
from syft.proto.grid.messages.association_messages_pb2 import (
    RespondAssociationRequestMessage as RespondAssociationRequestMessage_PB,
)
from syft.proto.grid.messages.association_messages_pb2 import (
    SendAssociationRequestMessage as SendAssociationRequestMessage_PB,
)

# relative
from ......core.common.serde.serializable import bind_protobuf


@bind_protobuf
@final
class SendAssociationRequestMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        source: AbstractNodeClient,
        target: AbstractNodeClient,
        address: Address,
        reply_to: Address,
        metadata: Dict[str, str],
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.source = source
        self.target = target
        self.metadata = metadata

    def _object2proto(self) -> SendAssociationRequestMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: SendAssociationRequestMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return SendAssociationRequestMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            source=serialize(self.source),
            target=serialize(self.target),
            reply_to=serialize(self.reply_to),
            metadata=self.metadata,
        )

    @staticmethod
    def _proto2object(
        proto: SendAssociationRequestMessage_PB,
    ) -> "SendAssociationRequestMessage":
        """Creates a SendAssociationRequestMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: SendAssociationRequestMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return SendAssociationRequestMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            source=_deserialize(blob=proto.source),
            target=_deserialize(blob=proto.target),
            reply_to=_deserialize(blob=proto.reply_to),
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

        return SendAssociationRequestMessage_PB


@bind_protobuf
@final
class ReceiveAssociationRequestMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        reply_to: Address,
        source: Client,
        target: Client,
        metadata: Dict[str, str],
        msg_id: Optional[UID] = None,
        response: Optional[str] = "",
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.metadata = metadata
        self.response = response
        self.source = source
        self.target = target

    def _object2proto(self) -> ReceiveAssociationRequestMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: ReceiveAssociationRequestMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return ReceiveAssociationRequestMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            response=self.response,
            reply_to=serialize(self.reply_to),
            metadata=self.metadata,
            source=serialize(self.source),
            target=serialize(self.target),
        )

    @staticmethod
    def _proto2object(
        proto: ReceiveAssociationRequestMessage_PB,
    ) -> "ReceiveAssociationRequestMessage":
        """Creates a ReceiveAssociationRequestMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: ReceiveAssociationRequestMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return ReceiveAssociationRequestMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            response=proto.response,
            reply_to=_deserialize(blob=proto.reply_to),
            metadata=proto.metadata,
            source=_deserialize(proto.source),
            target=_deserialize(proto.target),
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

        return ReceiveAssociationRequestMessage_PB


@bind_protobuf
@final
class RespondAssociationRequestMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        response: str,
        reply_to: Address,
        source: Client,
        target: Client,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.response = response
        self.source = source
        self.target = target

    def _object2proto(self) -> RespondAssociationRequestMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: RespondAssociationRequestMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return RespondAssociationRequestMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            source=serialize(self.source),
            target=serialize(self.target),
            response=self.response,
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: RespondAssociationRequestMessage_PB,
    ) -> "RespondAssociationRequestMessage":
        """Creates a RespondAssociationRequestMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: RespondAssociationRequestMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return RespondAssociationRequestMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            source=_deserialize(blob=proto.source),
            target=_deserialize(blob=proto.target),
            response=proto.response,
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

        return RespondAssociationRequestMessage_PB


@bind_protobuf
@final
class GetAssociationRequestMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        association_id: int,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.association_id = association_id

    def _object2proto(self) -> GetAssociationRequestMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: GetAssociationRequestMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetAssociationRequestMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            association_id=self.association_id,
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: GetAssociationRequestMessage_PB,
    ) -> "GetAssociationRequestMessage":
        """Creates a GetAssociationRequestMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: GetAssociationRequestMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetAssociationRequestMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            association_id=proto.association_id,
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

        return GetAssociationRequestMessage_PB


@bind_protobuf
@final
class GetAssociationRequestResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        metadata: Dict,
        source: Client,
        target: Client,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.metadata = metadata
        self.source = source
        self.target = target

    def _object2proto(self) -> GetAssociationRequestResponse_PB:
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
        return GetAssociationRequestResponse_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            metadata=self.metadata,
            source=serialize(self.source),
            target=serialize(self.target),
        )

    @staticmethod
    def _proto2object(
        proto: GetAssociationRequestResponse_PB,
    ) -> "GetAssociationRequestResponse":
        """Creates a SignalingOfferMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: SignalingOfferMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetAssociationRequestResponse(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            metadata=dict(proto.metadata),
            source=_deserialize(blob=proto.source),
            target=_deserialize(blob=proto.target),
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

        return GetAssociationRequestResponse_PB


@bind_protobuf
@final
class GetAssociationRequestsMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)

    def _object2proto(self) -> GetAssociationRequestsMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: GetAssociationRequestsMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetAssociationRequestsMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: GetAssociationRequestsMessage_PB,
    ) -> "GetAssociationRequestsMessage":
        """Creates a GetAssociationRequestsMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: GetAssociationRequestsMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetAssociationRequestsMessage(
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

        return GetAssociationRequestsMessage_PB


@bind_protobuf
@final
class GetAssociationRequestsResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        metadatas: List[Dict],
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.metadatas = metadatas

    def _object2proto(self) -> GetAssociationRequestsResponse_PB:
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
        msg = GetAssociationRequestsResponse_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
        )

        metadata_constructor = GetAssociationRequestsResponse_PB.metadata_container
        _ = [
            msg.metadatas.append(metadata_constructor(metadata=metadata))
            for metadata in self.metadatas
        ]
        return msg

    @staticmethod
    def _proto2object(
        proto: GetAssociationRequestsResponse_PB,
    ) -> "GetAssociationRequestsResponse":
        """Creates a SignalingOfferMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: SignalingOfferMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """
        return GetAssociationRequestsResponse(
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

        return GetAssociationRequestsResponse_PB


@bind_protobuf
@final
class DeleteAssociationRequestMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        association_id: int,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.association_id = association_id

    def _object2proto(self) -> DeleteAssociationRequestMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: DeleteAssociationRequestMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return DeleteAssociationRequestMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            association_id=self.association_id,
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: DeleteAssociationRequestMessage_PB,
    ) -> "DeleteAssociationRequestMessage":
        """Creates a DeleteAssociationRequestMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: DeleteAssociationRequestMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return DeleteAssociationRequestMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            association_id=proto.association_id,
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

        return DeleteAssociationRequestMessage_PB
