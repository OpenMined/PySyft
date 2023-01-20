# stdlib
import json
from typing import Dict
from typing import List as TypeList
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import SigningKey
from typing_extensions import final

# relative
from ...... import serialize
from ......proto.grid.messages.setup_messages_pb2 import (
    CreateInitialSetUpMessage as CreateInitialSetUpMessage_PB,
)
from ......proto.grid.messages.setup_messages_pb2 import (
    GetSetUpMessage as GetSetUpMessage_PB,
)
from ......proto.grid.messages.setup_messages_pb2 import (
    GetSetUpResponse as GetSetUpResponse_PB,
)
from ......proto.grid.messages.setup_messages_pb2 import (
    UpdateSetupMessage as UpdateSetupMessage_PB,
)
from ......proto.grid.messages.setup_messages_pb2 import (
    UpdateSetupResponse as UpdateSetupResponse_PB,
)
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.deserialize import _deserialize
from .....common.serde.serializable import serializable
from .....common.uid import UID
from .....io.address import Address


@serializable()
@final
class GetSetUpMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)

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


@serializable()
@final
class GetSetUpResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        content: Dict,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
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


@serializable()
@final
class CreateInitialSetUpMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        name: str,
        email: str,
        password: str,
        domain_name: str,
        budget: float,
        reply_to: Address,
        signing_key: SigningKey,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.name = name
        self.email = email
        self.password = password
        self.domain_name = domain_name
        self.budget = budget
        self.signing_key = signing_key

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
            name=self.name,
            email=self.email,
            password=self.password,
            domain_name=self.domain_name,
            budget=self.budget,
            reply_to=serialize(self.reply_to),
            signing_key=bytes(self.signing_key),
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
            name=proto.name,
            email=proto.email,
            password=proto.password,
            budget=proto.budget,
            domain_name=proto.domain_name,
            reply_to=_deserialize(blob=proto.reply_to),
            signing_key=SigningKey(proto.signing_key),
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


@serializable()
@final
class UpdateSetupMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        domain_name: str,
        description: str,
        daa: bool,
        contact: str,
        reply_to: Address,
        daa_document: Optional[bytes] = b"",
        tags: Optional[TypeList] = None,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.daa = daa
        self.contact = contact
        self.description = description
        self.domain_name = domain_name
        self.daa_document = daa_document
        self.tags = tags if tags is not None else []

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
            domain_name=self.domain_name,
            contact=self.contact,
            daa=self.daa,
            description=self.description,
            daa_document=self.daa_document,
            tags=self.tags,
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
            daa=proto.daa,
            contact=proto.contact,
            domain_name=proto.domain_name,
            description=proto.description,
            daa_document=proto.daa_document,
            tags=[tag for tag in proto.tags],
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


@serializable()
@final
class UpdateSetupResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        content: Dict,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
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
