# stdlib
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from oblv import OblvClient
from typing_extensions import final

# relative
from ...... import serialize
from ......proto.grid.messages.oblv_messages_pb2 import (
    CheckEnclaveConnectionMessage as CheckEnclaveConnectionMessage_PB,
)
from ......proto.grid.messages.oblv_messages_pb2 import (
    CreateKeyPairMessage as CreateKeyPairMessage_PB,
)
from ......proto.grid.messages.oblv_messages_pb2 import (
    CreateKeyPairResponse as CreateKeyPairResponse_PB,
)
from ......proto.grid.messages.oblv_messages_pb2 import (
    GetPublicKeyMessage as GetPublicKeyMessage_PB,
)
from ......proto.grid.messages.oblv_messages_pb2 import (
    GetPublicKeyResponse as GetPublicKeyResponse_PB,
)
from ......proto.grid.messages.oblv_messages_pb2 import (
    PublishDatasetMessage as PublishDatasetMessage_PB,
)
from ......proto.grid.messages.oblv_messages_pb2 import (
    PublishDatasetResponse as PublishDatasetResponse_PB,
)
from ......proto.grid.messages.oblv_messages_pb2 import (
    SyftOblvClient as SyftOblvClient_PB,
)
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.deserialize import _deserialize
from .....common.serde.serializable import serializable
from .....common.uid import UID
from .....io.address import Address
from .....tensor.autodp.phi_tensor import TensorWrappedPhiTensorPointer


@serializable()
@final
class SyftOblvClient:
    @classmethod
    def from_client(
        cls,
        input: OblvClient
    ):
        obj = SyftOblvClient()
        obj.token=input.token
        obj.oblivious_user_id=input.oblivious_user_id
        obj.cookies=input.cookies
        obj.headers=input.headers
        obj.timeout=input.timeout
        obj.verify_ssl=input.verify_ssl
        return obj

    def __init__(
        self,
        token: Optional[str]=None,
        oblivious_user_id: Optional[str]=None,
        cookies: Optional[dict]={},
        headers: Optional[dict]={},
        timeout: float = 20,
        verify_ssl: bool = True
    ):
        super().__init__()
        self.token=token
        self.oblivious_user_id=oblivious_user_id
        self.cookies=cookies
        self.headers=headers
        self.timeout=timeout
        self.verify_ssl=verify_ssl


    def _object2proto(self) -> SyftOblvClient_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: SyftOblvClient_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return SyftOblvClient_PB(
            token=self.token,
            oblivious_user_id=self.oblivious_user_id,
            cookies=self.cookies,
            headers=self.headers,
            timeout=self.timeout,
            verify_ssl=self.verify_ssl,
        )

    @staticmethod
    def _proto2object(
        proto: SyftOblvClient_PB,
    ) -> "SyftOblvClient":
        """Creates a SyftOblvClient from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: SyftOblvClient
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return SyftOblvClient(
            token=proto.token,
            oblivious_user_id=proto.oblivious_user_id,
            cookies=proto.cookies,
            headers=proto.headers,
            timeout=proto.timeout,
            verify_ssl=proto.verify_ssl,
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

        return SyftOblvClient_PB


@serializable()
@final
class CreateKeyPairMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)

    def _object2proto(self) -> CreateKeyPairMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: CreateKeyPairMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return CreateKeyPairMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: CreateKeyPairMessage_PB,
    ) -> "CreateKeyPairMessage":
        """Creates a CreateKeyPairMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: CreateKeyPairMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return CreateKeyPairMessage(
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

        return CreateKeyPairMessage_PB

@serializable()
@final
class CreateKeyPairResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        response: str,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.response = response

    def _object2proto(self) -> CreateKeyPairResponse_PB:
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
        return CreateKeyPairResponse_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            response=self.response,
        )

    @staticmethod
    def _proto2object(
        proto: CreateKeyPairResponse_PB,
    ) -> "CreateKeyPairResponse":
        """Creates a SignalingOfferMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: SignalingOfferMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return CreateKeyPairResponse(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            response=proto.response,
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

        return CreateKeyPairResponse_PB

@serializable()
@final
class GetPublicKeyMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)

    def _object2proto(self) -> GetPublicKeyMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: GetPublicKeyMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetPublicKeyMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: GetPublicKeyMessage_PB,
    ) -> "GetPublicKeyMessage":
        """Creates a GetPublicKeyMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: GetPublicKeyMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetPublicKeyMessage(
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

        return GetPublicKeyMessage_PB

@serializable()
@final
class GetPublicKeyResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        response: str = "",
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.response = response

    def _object2proto(self) -> GetPublicKeyResponse_PB:
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
        return GetPublicKeyResponse_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            response=self.response,
        )

    @staticmethod
    def _proto2object(
        proto: GetPublicKeyResponse_PB,
    ) -> "GetPublicKeyResponse":
        """Creates a SignalingOfferMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: SignalingOfferMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetPublicKeyResponse(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            response=proto.response,
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

        return GetPublicKeyResponse_PB

@serializable()
@final
class PublishDatasetMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        reply_to: Address,
        deployment_id: str,
        host_or_ip: str,
        port: int,
        protocol: str,
        client: SyftOblvClient,
        dataset_id: Union[str,TensorWrappedPhiTensorPointer] = "",
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.deployment_id = deployment_id
        self.client = client
        if type(dataset_id)==TensorWrappedPhiTensorPointer:
            self.dataset_id = dataset_id.id_at_location.to_string()
        else:
            self.dataset_id = dataset_id
        self.host_or_ip = host_or_ip
        self.protocol = protocol
        self.port = port

    def _object2proto(self) -> PublishDatasetMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: PublishDatasetMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return PublishDatasetMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            reply_to=serialize(self.reply_to),
            dataset_id=self.dataset_id,
            deployment_id=self.deployment_id,
            host_or_ip = self.host_or_ip,
            protocol = self.protocol,
            port = self.port,
            client=serialize(self.client),
        )

    @staticmethod
    def _proto2object(
        proto: PublishDatasetMessage_PB,
    ) -> "PublishDatasetMessage":
        """Creates a PublishDatasetMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: PublishDatasetMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return PublishDatasetMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            reply_to=_deserialize(blob=proto.reply_to),
            dataset_id=proto.dataset_id,
            deployment_id=proto.deployment_id,
            host_or_ip=proto.host_or_ip,
            protocol=proto.protocol,
            port=proto.port,
            client=_deserialize(blob=proto.client),
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

        return PublishDatasetMessage_PB

@serializable()
@final
class CheckEnclaveConnectionMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        reply_to: Address,
        deployment_id: str,
        client: SyftOblvClient,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.deployment_id = deployment_id
        self.client = client

    def _object2proto(self) -> CheckEnclaveConnectionMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: CheckEnclaveConnectionMessage_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return CheckEnclaveConnectionMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            reply_to=serialize(self.reply_to),
            deployment_id=self.deployment_id,
            client=serialize(self.client)
        )

    @staticmethod
    def _proto2object(
        proto: CheckEnclaveConnectionMessage_PB,
    ) -> "CheckEnclaveConnectionMessage":
        """Creates a CheckEnclaveConnectionMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: CheckEnclaveConnectionMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """
        
        return CheckEnclaveConnectionMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            reply_to=_deserialize(blob=proto.reply_to),
            deployment_id=proto.deployment_id,
            client = _deserialize(blob=proto.client)
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

        return CheckEnclaveConnectionMessage_PB


@serializable()
@final
class PublishDatasetResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        dataset_id: str = "",
        dataset_name: str = "",
        client: SyftOblvClient = None,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.client = client

    def _object2proto(self) -> PublishDatasetResponse_PB:
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
        return PublishDatasetResponse_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            dataset_id = self.dataset_id,
            dataset_name = self.dataset_name,
            client=serialize(self.client)
        )

    @staticmethod
    def _proto2object(
        proto: PublishDatasetResponse_PB,
    ) -> "PublishDatasetResponse":
        """Creates a SignalingOfferMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: SignalingOfferMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return PublishDatasetResponse(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            dataset_id = proto.dataset_id,
            dataset_name = proto.dataset_name,
            client = _deserialize(blob=proto.client)
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

        return PublishDatasetResponse_PB
