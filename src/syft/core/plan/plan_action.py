# stdlib
import functools
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# syft absolute
import syft as sy
# from syft.core.node.common.action.run_class_method_action import RunClassMethodAction
# syft relative
from ..common.serde import Serializable
from ...proto.core.plan.plan_action_pb2 import PlanAction as PlanAction_pb
from .plan_run_class_method_action import PlanRunClassMethodAction
from ..common.serde.serializable import bind_protobuf
# from ..node.common.action.run_class_method_action import RunClassMethodAction
from .plan_pointer import PlanPointer


@bind_protobuf
class PlanAction(Serializable):
    """
    When executing a RunClassMethodAction, a :class:`Node` will run a method defined
    by the action's path attribute on the object pointed at by _self and keep the returned
    value in its store.

    Attributes:
         path: the dotted path to the method to call
         _self: a pointer to the object which the method should be applied to.
         args: args to pass to the function. They should be pointers to objects
            located on the :class:`Node` that will execute the action.
         kwargs: kwargs to pass to the function. They should be pointers to objects
            located on the :class:`Node` that will execute the action.
    """

    def __init__(
        self,
        obj_type: str,
        action: PlanRunClassMethodAction,
    ):
        self.obj_type = obj_type
        self.action = action

    def _object2proto(self) -> PlanAction_pb:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: RunClassMethodAction_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """

        return PlanAction_pb(
            obj_type=self.obj_type,
            plan_run_class_method_action=self.action
        )

    @staticmethod
    def _proto2object(proto: PlanAction_pb) -> "PlanAction":
        """Creates a ObjectWithID from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of RunClassMethodAction
        :rtype: RunClassMethodAction

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return PlanAction(
            obj_type=proto.obj_type,
            action=proto.plan_run_class_method_action
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

        return PlanAction_pb
