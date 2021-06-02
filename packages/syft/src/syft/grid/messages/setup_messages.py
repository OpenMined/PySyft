# stdlib
import json
from typing import Dict
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from typing_extensions import final

# syft absolute
from syft import serialize
from syft.core.common.message import ImmediateSyftMessageWithReply
from syft.core.common.message import ImmediateSyftMessageWithoutReply
from syft.core.common.serde.deserialize import _deserialize
from syft.core.common.serde.serializable import bind_protobuf
from syft.core.common.uid import UID
from syft.core.io.address import Address
from syft.proto.grid.messages.setup_messages_pb2 import (
    CreateInitialSetUpMessage as CreateInitialSetUpMessage_PB,
)
from syft.proto.grid.messages.setup_messages_pb2 import (
    CreateInitialSetUpResponse as CreateInitialSetUpResponse_PB,
)
from syft.proto.grid.messages.setup_messages_pb2 import (
    GetSetUpMessage as GetSetUpMessage_PB,
)
from syft.proto.grid.messages.setup_messages_pb2 import (
    GetSetUpResponse as GetSetUpResponse_PB,
)
from syft.proto.grid.messages.setup_messages_pb2 import (
    UpdateSetupMessage as UpdateSetupMessage_PB,
)
from syft.proto.grid.messages.setup_messages_pb2 import (
    UpdateSetupResponse as UpdateSetupResponse_PB,
)


@bind_protobuf
@final
class GetSetUpMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        content: Dict,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.content = content

    def _object2proto(self) -> GetSetUpMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: GetSetUpMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetSetUpMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            content=json.dumps(self.content),
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: GetSetUpMessage_PB,
    ) -> "GetSetUpMessage":
        """Creates a GetSetUpMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: GetSetUpMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetSetUpMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            content=json.loads(proto.content),
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

        return GetSetUpMessage_PB


@bind_protobuf
@final
class GetSetUpResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        status_code: int,
        content: Dict,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.status_code = status_code
        self.content = content

    def _object2proto(self) -> GetSetUpResponse_PB:
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
        return GetSetUpResponse_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            status_code=self.status_code,
            content=json.dumps(self.content),
        )

    @staticmethod
    def _proto2object(
        proto: GetSetUpResponse_PB,
    ) -> "GetSetUpResponse":
        """Creates a SignalingOfferMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: SignalingOfferMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetSetUpResponse(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            status_code=proto.status_code,
            content=json.loads(proto.content),
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

        return GetSetUpResponse_PB


@bind_protobuf
@final
class CreateInitialSetUpMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        content: Dict,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.content = content

    def _object2proto(self) -> CreateInitialSetUpMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: CreateInitialSetUpMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return CreateInitialSetUpMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            content=json.dumps(self.content),
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: CreateInitialSetUpMessage_PB,
    ) -> "CreateInitialSetUpMessage":
        """Creates a CreateInitialSetUpMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: CreateInitialSetUpMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return CreateInitialSetUpMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            content=json.loads(proto.content),
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

        return CreateInitialSetUpMessage_PB


@bind_protobuf
@final
class CreateInitialSetUpResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        status_code: int,
        content: Dict,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.status_code = status_code
        self.content = content

    def _object2proto(self) -> CreateInitialSetUpResponse_PB:
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
        return CreateInitialSetUpResponse_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            status_code=self.status_code,
            content=json.dumps(self.content),
        )

    @staticmethod
    def _proto2object(
        proto: CreateInitialSetUpResponse_PB,
    ) -> "CreateInitialSetUpResponse":
        """Creates a SignalingOfferMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: SignalingOfferMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return CreateInitialSetUpResponse(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            status_code=proto.status_code,
            content=json.loads(proto.content),
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

        return CreateInitialSetUpResponse_PB


@bind_protobuf
@final
class UpdateSetupMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        content: Dict,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.content = content

    def _object2proto(self) -> UpdateSetupMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: UpdateSetupMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return UpdateSetupMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            content=json.dumps(self.content),
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: UpdateSetupMessage_PB,
    ) -> "UpdateSetupMessage":
        """Creates a UpdateSetupMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: UpdateSetupMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return UpdateSetupMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            content=json.loads(proto.content),
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

        return UpdateSetupMessage_PB


@bind_protobuf
@final
class UpdateSetupResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        status_code: int,
        content: Dict,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.status_code = status_code
        self.content = content

    def _object2proto(self) -> UpdateSetupResponse_PB:
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
        return UpdateSetupResponse_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            status_code=self.status_code,
            content=json.dumps(self.content),
        )

    @staticmethod
    def _proto2object(
        proto: UpdateSetupResponse_PB,
    ) -> "UpdateSetupResponse":
        """Creates a SignalingOfferMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: SignalingOfferMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return UpdateSetupResponse(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            status_code=proto.status_code,
            content=json.loads(proto.content),
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

        return UpdateSetupResponse_PB
