# DOs and Don's of this class:
# - Do NOT use absolute syft imports (i.e. import syft.core...) Use relative ones.
# - Do NOT put multiple imports on the same line (i.e. from <x> import a, b, c). Use separate lines
# - Do sort imports by length
# - Do group imports by where they come from

# stdlib
import json
from typing import Dict
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from typing_extensions import final

# syft relative
from .....decorators.syft_decorator_impl import syft_decorator
from .....proto.core.node.common.service.repr_service_pb2 import (
    CreateVMMessage as CreateVMMessage_PB,
)
from .....proto.core.node.common.service.repr_service_pb2 import (
    CreateVMResponseMessage as CreateVMResponseMessage_PB,
)
from ....common.message import ImmediateSyftMessageWithReply
from ....common.message import ImmediateSyftMessageWithoutReply
from ....common.serde.deserialize import _deserialize
from ....common.uid import UID
from ....io.address import Address


@final
class CreateVMMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        settings: Dict,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.settings = settings

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> CreateVMMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: CreateVMMessage_PB

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """

        return CreateVMMessage_PB(
            msg_id=self.id.serialize(),
            address=self.address.serialize(),
            settings=json.dumps(self.settings),
            reply_to=self.reply_to.serialize(),
        )

    @staticmethod
    def _proto2object(proto: CreateVMMessage_PB) -> "CreateVMMessage":
        """Creates a CreateVMMessage from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of CreateVMMessage
        :rtype: CreateVMMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return CreateVMMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            settings=json.loads(proto.settings),
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
        with the type of this class attached to it. See the MetaSerializable class for details.

        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType

        """

        return CreateVMMessage_PB


@final
class CreateVMResponseMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        vm_address: Address,
        success: bool,
        msg: Optional[str],
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.vm_address = vm_address
        self.success = success
        self.msg = msg

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> CreateVMResponseMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: CreateVMResponseMessage

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """

        return CreateVMResponseMessage_PB(
            msg_id=self.id.serialize(),
            address=self.address.serialize(),
            vm_address=self.vm_address.serialize(),
            success=self.success,
            msg=self.msg,
        )

    @staticmethod
    def _proto2object(proto: CreateVMResponseMessage_PB) -> "CreateVMResponseMessage":
        """Creates a CreateVMResponseMessage from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of CreateVMResponseMessage
        :rtype: CreateVMResponseMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return CreateVMResponseMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            vm_address=_deserialize(blob=proto.vm_address),
            success=_deserialize(blob=proto.success),
            msg=_deserialize(blob=proto.msg),
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
        with the type of this class attached to it. See the MetaSerializable class for details.

        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType

        """

        return CreateVMResponseMessage_PB
