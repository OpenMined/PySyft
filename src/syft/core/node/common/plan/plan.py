# stdlib
import re
import sys
from typing import List

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft absolute
from syft.core.common.object import Serializable
from syft.core.node.common.action.common import Action
from syft.proto.core.node.common.action.action_pb2 import Action as Action_PB
from syft.proto.core.node.common.plan.plan_pb2 import Plan as Plan_PB
from ...abstract.node import AbstractNode
from nacl.signing import VerifyKey

CAMEL_TO_SNAKE_PAT = re.compile(r"(?<!^)(?=[A-Z])")


class Plan(Serializable):
    def __init__(self, actions: List[Action]):
        self.actions = actions

    def execute(self, node: AbstractNode, verify_key: VerifyKey):
        for a in self.actions:
            a.execute_action(node, verify_key)

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

        return Plan_PB

    def _object2proto(self) -> Plan_PB:
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

        def camel_to_snake(s):
            return CAMEL_TO_SNAKE_PAT.sub("_", s).lower()

        actions_pb = [
            Action_PB(
                obj_type=".".join([action.__module__, action.__class__.__name__]),
                **{camel_to_snake(action.__class__.__name__): action.serialize()}
            )
            for action in self.actions
        ]

        return Plan_PB(actions=actions_pb)

    @staticmethod
    def _proto2object(proto: Plan_PB) -> "GetObjectAction":
        """Creates a ObjectWithID from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of GetObjectAction
        :rtype: GetObjectAction

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """
        actions = []

        for action_proto in proto.actions:
            module, cls_name = action_proto.obj_type.rsplit(".", 1)
            action_cls = getattr(sys.modules[module], cls_name)

            # protobuf does no inheritance, so we wrap action subclasses
            # in the main action class.
            inner_action = getattr(action_proto, action_proto.WhichOneof("action"))
            actions.append(action_cls._proto2object(inner_action))

        return Plan(actions=actions)
