# external class imports
from nacl.signing import VerifyKey
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft imports
from ....common.message import ImmediateSyftMessageWithoutReply
from .....decorators.syft_decorator_impl import syft_decorator
from ....common.serde.deserialize import _deserialize
from .common import ImmediateActionWithReply
from ...abstract.node import AbstractNode

# syft proto imports
from .....proto.core.node.common.action.get_object_pb2 import (
    GetObjectResponseMessage as GetObjectResponseMessage_PB,
)
from .....proto.core.node.common.action.get_object_pb2 import (
    GetObjectAction as GetObjectAction_PB,
)


class GetObjectResponseMessage(ImmediateSyftMessageWithoutReply):
    def __init__(self, obj, address, msg_id=None):
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
        return GetObjectResponseMessage_PB(
            msg_id=self.id.serialize(),
            address=self.address.serialize(),
            obj=self.obj.serialize(),
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
        """ Return the type of protobuf object which stores a class of this type

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
    def __init__(self, obj_id, address, reply_to, msg_id=None):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.obj_id = obj_id

    def execute_action(
        self, node: AbstractNode, verify_key: VerifyKey
    ) -> ImmediateSyftMessageWithoutReply:

        storeable_object = node.store[self.obj_id]

        if verify_key not in storeable_object.read_permissions:
            raise Exception(
                "You do not have permission to .get() this tensor. Please submit a request."
            )

        obj = storeable_object.data
        msg = GetObjectResponseMessage(obj=obj, address=self.reply_to, msg_id=None)

        # TODO: send EventualActionWithoutReply to delete the object at the node's
        # convenience instead of definitely having to delete it now
        del node.store[self.obj_id]
        return msg

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
        """ Return the type of protobuf object which stores a class of this type

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
