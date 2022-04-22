# stdlib
from typing import Dict
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# syft absolute
import syft as sy

# relative
from ..... import lib
from .....proto.core.node.common.action.get_enum_attribute_pb2 import (
    GetEnumAttributeAction as GetEnumAttributeAction_PB,
)
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ....io.address import Address
from ....store.storeable_object import StorableObject
from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply
from .run_class_method_action import RunClassMethodAction


@serializable()
class EnumAttributeAction(ImmediateActionWithoutReply):
    def __init__(
        self,
        path: str,
        id_at_location: UID,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address, msg_id=msg_id)
        self.id_at_location = id_at_location
        self.path = path

    def intersect_keys(
        self,
        left: Dict[VerifyKey, Optional[UID]],
        right: Dict[VerifyKey, Optional[UID]],
    ) -> Dict[VerifyKey, Optional[UID]]:
        return RunClassMethodAction.intersect_keys(left, right)

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        enum_attribute = node.lib_ast.query(self.path)
        result = enum_attribute.solve_get_enum_attribute().value
        result = lib.python.primitive_factory.PrimitiveFactory.generate_primitive(
            value=result, id=self.id_at_location
        )

        result = StorableObject(
            id=self.id_at_location,
            data=result,
        )

        node.store[self.id_at_location] = result

    def _object2proto(self) -> GetEnumAttributeAction_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: GetOrSetPropertyAction_PB
        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """

        return GetEnumAttributeAction_PB(
            path=self.path,
            id_at_location=sy.serialize(self.id_at_location),
            address=sy.serialize(self.address),
            msg_id=sy.serialize(self.id),
        )

    @staticmethod
    def _proto2object(
        proto: GetEnumAttributeAction_PB,
    ) -> "EnumAttributeAction":
        """Creates a ObjectWithID from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of GetOrSetPropertyAction
        :rtype: GetOrSetPropertyAction
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """
        return EnumAttributeAction(
            path=proto.path,
            id_at_location=sy.deserialize(blob=proto.id_at_location),
            address=sy.deserialize(blob=proto.address),
            msg_id=sy.deserialize(blob=proto.msg_id),
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

        return GetEnumAttributeAction_PB
