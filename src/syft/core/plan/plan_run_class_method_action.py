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
from ...proto.core.node.common.action.plan_run_class_method_pb2 import PlanRunClassMethodAction as PlanRunClassMethodAction_PB
from ..common.serde.deserialize import _deserialize
from ..common.serde.serializable import bind_protobuf
# from ..node.common.action.run_class_method_action import RunClassMethodAction
from .plan_pointer import PlanPointer


@bind_protobuf
class PlanRunClassMethodAction():
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
        method_action: str,
        seq_id: str,
        _self: Any = None,
        args: List[Any] = None,
        kwargs: Dict[Any, Any] = None,
    ):
        self.path = method_action
        self.args = args
        self._self = _self
        self.kwargs = kwargs
        self.seq_id = seq_id

    def _object2proto(self) -> PlanRunClassMethodAction_PB:
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

        return PlanRunClassMethodAction_PB(
            path=self.path,
            _self=self._self._object2proto(),
            args=list(map(lambda x: x._object2proto(), self.args)),
            kwargs={k: v._object2proto() for k, v in self.kwargs.items()},
            seq_id=self.seq_id,
        )

    @staticmethod
    def _proto2object(proto: PlanRunClassMethodAction_PB) -> "PlanRunClassMethodAction":
        """Creates a ObjectWithID from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of RunClassMethodAction
        :rtype: RunClassMethodAction

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return PlanRunClassMethodAction(
            method_action=proto.path,
            seq_id=proto.seq_id,
            _self=PlanPointer._proto2object(blob=proto._self),
            args=list(map(lambda x: PlanPointer._proto2object(blob=x), proto.args)),
            kwargs={k: PlanPointer._proto2object(blob=v) for k, v in proto.kwargs.items()},
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

        return PlanRunClassMethodAction_PB

    def remap_input(self, current_input: Any, new_input: Any) -> None:
        """Redefines some of the arguments, and possibly the _self of the function"""
        if self._self.id_at_location == current_input.id_at_location:
            self._self = new_input

        for i, arg in enumerate(self.args):
            if arg.id_at_location == current_input.id_at_location:
                self.args[i] = new_input

        for k, v in self.kwargs.items():
            if v.id_at_location == current_input.id_at_location:
                self.kwargs[k] = new_input
