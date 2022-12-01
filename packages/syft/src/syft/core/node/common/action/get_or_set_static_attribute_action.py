# stdlib
from enum import Enum
from typing import Any
from typing import Dict
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# relative
from ..... import lib
from .....proto.core.node.common.action.get_set_static_attribute_pb2 import (
    GetSetStaticAttributeAction as GetSetStaticAttributeAction_PB,
)
from ....common.serde.deserialize import _deserialize as deserialize
from ....common.serde.serializable import serializable
from ....common.serde.serialize import _serialize as serialize
from ....common.uid import UID
from ....io.address import Address
from ....store.storeable_object import StorableObject
from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply
from .run_class_method_action import RunClassMethodAction


class StaticAttributeAction(Enum):
    SET = 1
    GET = 2


@serializable()
class GetSetStaticAttributeAction(ImmediateActionWithoutReply):
    def __init__(
        self,
        path: str,
        id_at_location: UID,
        address: Address,
        action: StaticAttributeAction,
        msg_id: Optional[UID] = None,
        set_arg: Optional[Any] = None,
    ):
        super().__init__(address, msg_id=msg_id)
        self.path = path
        self.id_at_location = id_at_location
        self.action = action
        self.set_arg = set_arg

    def intersect_keys(
        self,
        left: Dict[VerifyKey, Optional[UID]],
        right: Dict[VerifyKey, Optional[UID]],
    ) -> Dict[VerifyKey, Optional[UID]]:
        return RunClassMethodAction.intersect_keys(left, right)

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        static_attribute_solver = node.lib_ast.query(self.path)

        if self.action == StaticAttributeAction.SET:
            if self.set_arg is None:
                raise ValueError("MAKE PROPER SCHEMA")

            resolved_arg = node.store.get(key=self.set_arg.id_at_location)
            result = static_attribute_solver.solve_set_value(resolved_arg)
        elif self.action == StaticAttributeAction.GET:
            result = static_attribute_solver.solve_get_value()
        else:
            raise ValueError(f"{self.action} not a valid action!")

        if lib.python.primitive_factory.isprimitive(value=result):
            result = lib.python.primitive_factory.PrimitiveFactory.generate_primitive(
                value=result, id=self.id_at_location
            )
        else:
            if hasattr(result, "id"):
                try:
                    if hasattr(result, "_id"):
                        # set the underlying id
                        result._id = self.id_at_location
                    else:
                        result.id = self.id_at_location

                    if result.id != self.id_at_location:
                        raise AttributeError("IDs don't match")
                except AttributeError as e:
                    err = f"Unable to set id on result {type(result)}. {e}"
                    raise Exception(err)

        if not isinstance(result, StorableObject):
            result = StorableObject(
                id=self.id_at_location,
                data=result,
            )

        node.store[self.id_at_location] = result

    def _object2proto(self) -> GetSetStaticAttributeAction_PB:
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

        # this is ugly
        if self.set_arg is not None:
            return GetSetStaticAttributeAction_PB(
                path=self.path,
                id_at_location=serialize(self.id_at_location),
                address=serialize(self.address),
                msg_id=serialize(self.id),
                action=self.action.value,
                set_arg=serialize(self.set_arg),
            )
        else:
            return GetSetStaticAttributeAction_PB(
                path=self.path,
                id_at_location=serialize(self.id_at_location),
                address=serialize(self.address),
                msg_id=serialize(self.id),
                action=self.action.value,
            )

    @staticmethod
    def _proto2object(
        proto: GetSetStaticAttributeAction_PB,
    ) -> "GetSetStaticAttributeAction":
        """Creates a ObjectWithID from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of GetOrSetPropertyAction
        :rtype: GetOrSetPropertyAction
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """
        return GetSetStaticAttributeAction(
            path=proto.path,
            id_at_location=deserialize(blob=proto.id_at_location),
            address=deserialize(blob=proto.address),
            msg_id=deserialize(blob=proto.msg_id),
            set_arg=deserialize(blob=proto.set_arg)
            if proto.HasField("set_arg")
            else None,
            action=StaticAttributeAction(proto.action),
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

        return GetSetStaticAttributeAction_PB
