# stdlib
import re
import sys
from typing import Any
from typing import List
from typing import Tuple
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# syft absolute
from syft.core.common.object import Serializable
from syft.core.common.serde.serializable import bind_protobuf
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.action.common import Action
from syft.core.node.common.util import listify
from syft.core.pointer.pointer import Pointer
from syft.proto.core.node.common.action.action_pb2 import Action as Action_PB
from syft.proto.core.node.common.plan.plan_pb2 import Plan as Plan_PB

# from ...abstract.node import AbstractNode
# from ..util import listify

CAMEL_TO_SNAKE_PAT = re.compile(r"(?<!^)(?=[A-Z])")


@bind_protobuf
class Plan(Serializable):
    """
    A plan is a collection of actions, plus some variable inputs, that together form a computation graph.

    Attributes:
        actions: list of actions
        inputs: Pointers to the inputs. Defaults to None.
    """

    def __init__(
        self, actions: List[Action], inputs: Union[Pointer, List[Pointer], None] = None
    ):
        self.actions = actions
        self.inputs: List[Pointer] = listify(inputs)

    def __call__(
        self, node: AbstractNode, verify_key: VerifyKey, *args: Tuple[Any]
    ) -> None:
        """
        1) For all pointers that were passed into the init as `inputs`, this method
           replaces those pointers in self.actions by the pointers passed in as *args.
        2) Executes the actions in self.actions one by one

        *While this function requires `node` and `verify_key` as inputs, during remote
        execution, passing these is handled in `RunClassMethodAction`*

        *Note that this method will receive *args as pointers during execution. Normally,
        pointers are resolved during `RunClassMethodAction.execute()`, but not for plans,
        as they need to operate on the pointer to enable remapping of the inputs.*
        Args:
            *args: the new inputs for the plan, passed as pointers
        """
        inputs = listify(args)

        # this is pretty cumbersome, we are searching through all actions to check
        # if we need to redefine some of their attributes that are inputs in the
        # graph of actions
        for i, (current_input, new_input) in enumerate(zip(self.inputs, inputs)):
            for a in self.actions:
                if hasattr(a, "remap_input"):
                    a.remap_input(current_input, new_input)

            # redefine the inputs of the plan
            self.inputs[i] = new_input

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

        def camel_to_snake(s: str) -> str:
            return CAMEL_TO_SNAKE_PAT.sub("_", s).lower()

        actions_pb = [
            Action_PB(
                obj_type=".".join([action.__module__, action.__class__.__name__]),
                **{camel_to_snake(action.__class__.__name__): action.serialize()}
            )
            for action in self.actions
        ]
        inputs_pb = [inp._object2proto() for inp in self.inputs]

        return Plan_PB(actions=actions_pb, inputs=inputs_pb)

    @staticmethod
    def _proto2object(proto: Plan_PB) -> "Plan":
        """Creates a ObjectWithID from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of Plan
        :rtype: Plan

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

        inputs = [
            Pointer._proto2object(pointer_proto) for pointer_proto in proto.inputs
        ]

        return Plan(actions=actions, inputs=inputs)
