# stdlib
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# syft relative
from .....decorators.syft_decorator_impl import syft_decorator
from .....proto.core.node.common.action.get_object_pb2 import (
    GetObjectAction as GetObjectAction_PB,
)
from .....proto.core.node.common.action.get_object_pb2 import (
    GetObjectResponseMessage as GetObjectResponseMessage_PB,
)
from .....proto.core.store.store_object_pb2 import StorableObject as StorableObject_PB
from ....common.message import ImmediateSyftMessageWithoutReply
from ....common.serde.deserialize import _deserialize
from ....common.uid import UID
from ....io.address import Address
from ....store.storeable_object import StorableObject
from ...abstract.node import AbstractNode
from ..service.auth import AuthorizationException
from .common import ImmediateActionWithReply


class GetObjectResponseMessage(ImmediateSyftMessageWithoutReply):
    """
    GetObjectResponseMessages are the type of messages that are sent in reponse to a
    :class:`GetObjectAction`. They contain the object that was asked for.

    Attributes:
         obj: the object being sent back to the asker.
    """

    def __init__(
        self, obj: StorableObject, address: Address, msg_id: Optional[UID] = None
    ) -> None:
        super().__init__(address=address, msg_id=msg_id)
        self.obj = obj

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> GetObjectResponseMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: GetObjectResponseMessage_PB

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """

        ser = self.obj.serialize()
        # TODO: Fix this hack
        # we need to check if the serialize chain creates a storable if not
        # we need to go use the serializable_wrapper_type
        # this is because klasses have a different def serialize to normal serializables
        # which checks for the serializable_wrapper_type and uses it
        if not isinstance(ser, StorableObject_PB):
            if hasattr(self.obj, "serializable_wrapper_type"):
                obj = self.obj.serializable_wrapper_type(value=self.obj)  # type: ignore
                ser = obj.serialize()
            else:
                raise Exception(f"Cannot send {type(self.obj)} as StorableObject")

        return GetObjectResponseMessage_PB(
            msg_id=self.id.serialize(),
            address=self.address.serialize(),
            obj=ser,
        )

    @staticmethod
    def _proto2object(proto: GetObjectResponseMessage_PB) -> "GetObjectResponseMessage":
        """Creates a ObjectWithID from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of GetObjectResponseMessage
        :rtype: GetObjectResponseMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetObjectResponseMessage(
            obj=_deserialize(blob=proto.obj),
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
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

        return GetObjectResponseMessage_PB


class GetObjectAction(ImmediateActionWithReply):
    """
    This kind of action is used when a Node wants to get an object located on another Node.

    The Node receiving this action first check that the asker does have the permission to
    fetch the object he asked for. If it's the case, a :class:`GetObjectResponseMessage`
    containing the object is sent back to the asker.

    Attributes:
         obj_id: the id of the object asked for.
    """

    def __init__(
        self,
        obj_id: UID,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        self.obj_id = obj_id

        # the logger needs self.obj_id to be set already - so we call this later
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)

    def execute_action(
        self, node: AbstractNode, verify_key: VerifyKey
    ) -> ImmediateSyftMessageWithoutReply:

        storeable_object = node.store[self.obj_id]

        if verify_key not in storeable_object.read_permissions:
            raise AuthorizationException(
                "You do not have permission to .get() this tensor. Please submit a request."
            )

        obj = storeable_object.data
        msg = GetObjectResponseMessage(obj=obj, address=self.reply_to, msg_id=None)

        # TODO: send EventualActionWithoutReply to delete the object at the node's
        # convenience instead of definitely having to delete it now
        del node.store[self.obj_id]
        return msg

    @property
    def pprint(self) -> str:
        return f"GetObjectAction({self.obj_id})"

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> GetObjectAction_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: ObjectWithID_PB

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetObjectAction_PB(
            obj_id=self.obj_id.proto(),
            msg_id=self.id.proto(),
            address=self.address.proto(),
            reply_to=self.reply_to.proto(),
        )

    @staticmethod
    def _proto2object(proto: GetObjectAction_PB) -> "GetObjectAction":
        """Creates a ObjectWithID from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of GetObjectAction
        :rtype: GetObjectAction

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetObjectAction(
            obj_id=_deserialize(blob=proto.obj_id),
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
        with the type of this class attached to it. See the MetaSerializable class for details.

        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType

        """

        return GetObjectAction_PB
