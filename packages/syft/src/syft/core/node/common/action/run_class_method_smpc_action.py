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
from syft import serialize
from syft.core.plan.plan import Plan

# relative
from ..... import deserialize
from ..... import lib
from ..... import serialize
from .....logger import critical
from .....logger import traceback_and_raise
from .....logger import warning
from .....proto.core.node.common.action.run_class_method_smpc_pb2 import (
    RunClassMethodSMPCAction as RunClassMethodSMPCAction_PB,
)
from .....util import inherit_tags
from ....common.serde.serializable import bind_protobuf
from ....common.uid import UID
from ....io.address import Address
from ....store.storeable_object import StorableObject
from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply


@bind_protobuf
class RunClassMethodSMPCAction(ImmediateActionWithoutReply):
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
        path: str,
        _self: Any,
        args: List[Any],
        kwargs: Dict[Any, Any],
        id_at_location: UID,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        self.path = path
        self._self = _self
        self.args = args
        self.kwargs = kwargs
        self.id_at_location = id_at_location
        # logging needs .path to exist before calling
        # this which is why i've put this super().__init__ down here
        super().__init__(address=address, msg_id=msg_id)

    @staticmethod
    def intersect_keys(
        left: Dict[VerifyKey, UID], right: Dict[VerifyKey, UID]
    ) -> Dict[VerifyKey, UID]:
        # get the intersection of the dict keys, the value is the request_id
        # if the request_id is different for some reason we still want to keep it,
        # so only intersect the keys and then copy those over from the main dict
        # into a new one
        intersection = set(left.keys()).intersection(right.keys())
        # left and right have the same keys
        return {k: left[k] for k in intersection}

    @property
    def pprint(self) -> str:
        return f"RunClassMethodSMPCAction({self.path})"

    def __repr__(self) -> str:
        method_name = self.path.split(".")[-1]
        self_name = self._self.class_name
        arg_names = ",".join([a.class_name for a in self.args])
        kwargs_names = ",".join([f"{k}={v.class_name}" for k, v in self.kwargs.items()])
        return f"RunClassMethodSMPCAction {self_name}.{method_name}({arg_names}, {kwargs_names})"

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        # relative
        from .smpc_action_message import SMPCActionMessage

        method = node.lib_ast(self.path)
        resolved_self = node.store.get_object(key=self._self.id_at_location)

        if resolved_self is None:
            critical(
                f"execute_action on {self.path} failed due to missing object"
                + f" at: {self._self.id_at_location}"
            )
            return
        result_read_permissions = resolved_self.read_permissions

        resolved_args = list()
        tag_args = []
        for arg in self.args:
            r_arg = node.store[arg.id_at_location]
            result_read_permissions = self.intersect_keys(
                result_read_permissions, r_arg.read_permissions
            )
            resolved_args.append(r_arg.data)
            tag_args.append(r_arg)

        resolved_kwargs = {}
        tag_kwargs = {}
        for arg_name, arg in self.kwargs.items():
            r_arg = node.store[arg.id_at_location]
            result_read_permissions = self.intersect_keys(
                result_read_permissions, r_arg.read_permissions
            )
            resolved_kwargs[arg_name] = r_arg.data
            tag_kwargs[arg_name] = r_arg

        (
            upcasted_args,
            upcasted_kwargs,
        ) = lib.python.util.upcast_args_and_kwargs(resolved_args, resolved_kwargs)

        method_name = self.path.split(".")[-1]

        seed_id_locations = resolved_kwargs.get("seed_id_locations", None)
        if seed_id_locations is None:
            raise ValueError(
                "Expected 'seed_id_locations' to be in the kwargs to generate id_at_location in a deterministic matter"
            )

        resolved_kwargs.pop("seed_id_locations")
        actions_generator = SMPCActionMessage.get_action_generator_from_op(method_name)
        args_id = [arg.id_at_location for arg in self.args]

        kwargs = {
            "seed_id_locations": seed_id_locations,
            "node": node,
        }  # TODO: the seed should be sent by the orchestrator

        # Get the list of actions to be run
        actions = actions_generator(self._self.id_at_location, *args_id, **kwargs)  # type: ignore
        actions = SMPCActionMessage.filter_actions_after_rank(
            resolved_self.data, actions
        )

        client = node.get_client()
        for action in actions:
            client.send_immediate_msg_without_reply(msg=action)

    def _object2proto(self) -> RunClassMethodSMPCAction_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: RunClassMethodSMPCAction_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """

        return RunClassMethodSMPCAction_PB(
            path=self.path,
            _self=serialize(self._self),
            args=list(map(lambda x: serialize(x), self.args)),
            kwargs={k: serialize(v) for k, v in self.kwargs.items()},
            id_at_location=serialize(self.id_at_location),
            address=serialize(self.address),
            msg_id=serialize(self.id),
        )

    @staticmethod
    def _proto2object(proto: RunClassMethodSMPCAction_PB) -> "RunClassMethodSMPCAction":
        """Creates a ObjectWithID from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of RunClassMethodSMPCAction
        :rtype: RunClassMethodSMPCAction

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return RunClassMethodSMPCAction(
            path=proto.path,
            _self=deserialize(blob=proto._self),
            args=list(map(lambda x: deserialize(blob=x), proto.args)),
            kwargs={k: deserialize(blob=v) for k, v in proto.kwargs.items()},
            id_at_location=deserialize(blob=proto.id_at_location),
            address=deserialize(blob=proto.address),
            msg_id=deserialize(blob=proto.msg_id),
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

        return RunClassMethodSMPCAction_PB
