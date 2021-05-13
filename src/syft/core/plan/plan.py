"""
This is the main Plan class which is responsible for containing a list of Actions
which can be serialized, deserialized, and executed by substituting the run time
pointers with the original traced pointers and replaying the actions against a node.
"""
# stdlib
import re
import sys
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# syft relative
from ... import serialize
from ...logger import traceback_and_raise
from ...proto.core.node.common.action.action_pb2 import Action as Action_PB
from ...proto.core.plan.plan_pb2 import Plan as Plan_PB
from ..common.object import Serializable
from ..common.serde.serializable import bind_protobuf
from ..node.abstract.node import AbstractNode
from ..node.common import client
from ..node.common.action.common import Action
from ..node.common.util import listify
from ..pointer.pointer import Pointer
from ..store.storeable_object import StorableObject

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
        self,
        actions: Union[List[Action], None] = None,
        inputs: Union[Dict[str, Pointer], None] = None,
        outputs: Union[Pointer, List[Pointer], None] = None,
        i2o_map: Union[Dict[str, int], None] = None,
        code: Optional[str] = None,
        max_calls: Optional[int] = None,
    ):
        """
        Initialize the Plan with actions, inputs and outputs
        """
        self.actions: List[Action] = listify(actions)
        self.inputs: Dict[str, Pointer] = inputs if inputs is not None else dict()
        self.outputs: List[Pointer] = listify(outputs)
        self.i2o_map: Dict[str, int] = i2o_map if i2o_map is not None else dict()
        self.code = code
        self.max_calls = max_calls
        self.n_calls = 0

    def __call__(
        self,
        node: Optional[AbstractNode] = None,
        verify_key: VerifyKey = None,
        **kwargs: Dict[str, Any],
    ) -> List[StorableObject]:
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

        self.n_calls += 1

        # this is pretty cumbersome, we are searching through all actions to check
        # if we need to redefine some of their attributes that are inputs in the
        # graph of actions
        if node is None:
            return self.execute_locally(**kwargs)

        new_inputs: Dict[str, Pointer] = {}
        for k, current_input in self.inputs.items():
            new_input = kwargs[k]
            if not issubclass(type(new_input), Pointer):
                traceback_and_raise(
                    f"Calling Plan without a Pointer. {k} == {type(new_input)} "
                )
            for a in self.actions:
                if hasattr(a, "remap_input"):
                    a.remap_input(current_input, new_input)  # type: ignore

            # redefine the inputs of the plan
            new_inputs[k] = new_input  # type: ignore
        self.inputs = new_inputs

        for a in self.actions:
            a.execute_action(node, verify_key)

        for k, v in self.i2o_map.items():
            self.outputs[v] = self.inputs[k]

        if len(self.outputs):
            resolved_outputs = []
            for arg in self.outputs:
                r_arg = node.store[arg.id_at_location]
                resolved_outputs.append(r_arg.data)
            return resolved_outputs
        else:
            return []

    def __repr__(self) -> str:
        obj_str = "Plan"

        allowed, remaining = (
            (self.max_calls, self.max_calls - self.n_calls)
            if self.max_calls is not None
            else ("not defined", "not defined")
        )

        ex_str = f"Allowed executions:\t{allowed}\nRemaining executions:\t{remaining}"

        inp_str = "Inputs:\n"
        inp_str += "\n".join(
            [f"\t\t{k}:\t{v.__class__.__name__}" for k, v in self.inputs.items()]
        )

        act_str = f"Actions:\n\t\t{len(self.actions)} Actions"

        out_str = "Outputs:\n"
        out_str += "\n".join([f"\t\t{o.__class__.__name__}" for o in self.outputs])

        plan_str = "Plan code:\n"
        plan_str += f'"""\n{self.code}\n"""' if self.code is not None else ""

        return f"{obj_str}\n{ex_str}\n{inp_str}\n{act_str}\n{out_str}\n\n{plan_str}"

    def execute_locally(self, **kwargs: Any) -> List[StorableObject]:
        """Execute a plan by sending it to a virtual machine and calling execute on the pointer.
        This is a workaround until we have a way to execute plans locally.
        """
        # prevent circular dependency
        # syft relative
        from ...core.node.vm.vm import VirtualMachine  # noqa: F401

        alice = VirtualMachine(name="plan_executor")
        alice_client: client.Client = alice.get_client()
        self_ptr = self.send(alice_client)  # type: ignore
        out = self_ptr(**kwargs)
        return out.get()

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
            """Convert CamelCase classes to snake case for matching protobuf names"""
            return CAMEL_TO_SNAKE_PAT.sub("_", s).lower()

        actions_pb = [
            Action_PB(
                obj_type=".".join([action.__module__, action.__class__.__name__]),
                **{camel_to_snake(action.__class__.__name__): serialize(action)},
            )
            for action in self.actions
        ]
        inputs_pb = {k: v._object2proto() for k, v in self.inputs.items()}
        outputs_pb = [out._object2proto() for out in self.outputs]
        i2o_map_pb = self.i2o_map

        return Plan_PB(
            actions=actions_pb, inputs=inputs_pb, outputs=outputs_pb, i2o_map=i2o_map_pb
        )

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

        inputs = {k: Pointer._proto2object(proto.inputs[k]) for k in proto.inputs}
        outputs = [
            Pointer._proto2object(pointer_proto) for pointer_proto in proto.outputs
        ]
        i2o_map = proto.i2o_map

        return Plan(actions=actions, inputs=inputs, outputs=outputs, i2o_map=i2o_map)
