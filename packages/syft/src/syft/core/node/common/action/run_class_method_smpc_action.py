# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# syft absolute
import syft as sy

# relative
from ..... import lib
from ..... import logger
from .....logger import traceback_and_raise
from .....proto.core.node.common.action.run_class_method_smpc_pb2 import (
    RunClassMethodSMPCAction as RunClassMethodSMPCAction_PB,
)
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ....io.address import Address
from ....store.storeable_object import StorableObject
from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply
from .greenlets_switch import retrieve_object

if TYPE_CHECKING:
    # relative
    from .smpc_action_message import SMPCActionMessage


@serializable()
class RunClassMethodSMPCAction(ImmediateActionWithoutReply):
    """
    When executing a RunClassMethodSMPCAction, a list of SMPCActionMessages is sent to the
    to the same node (using a rabbitMQ)

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
        self_name = self._self.__class__.__name__
        arg_names = ",".join([a.__class__.__name__ for a in self.args])
        kwargs_names = ",".join(
            [f"{k}={v.__class__.__name__}" for k, v in self.kwargs.items()]
        )
        return f"RunClassMethodSMPCAction {self_name}.{method_name}({arg_names}, {kwargs_names})"

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        # relative
        from . import smpc_action_functions
        from ..... import Tensor
        from .smpc_action_message import SMPCActionMessage
        from .smpc_action_seq_batch_message import SMPCActionSeqBatchMessage

        resolved_self = retrieve_object(node, self._self.id_at_location, self.path)

        result_read_permissions = resolved_self.read_permissions

        resolved_args = list()
        tag_args = []
        for arg in self.args:
            r_arg = retrieve_object(node, arg.id_at_location, self.path)

            # TODO: Think of a way to free the memory
            # del node.store[arg.id_at_location]
            result_read_permissions = self.intersect_keys(
                result_read_permissions, r_arg.read_permissions
            )
            resolved_args.append(r_arg.data)
            tag_args.append(r_arg)

        resolved_kwargs = {}
        tag_kwargs = {}
        for arg_name, arg in self.kwargs.items():
            r_arg = retrieve_object(node, arg.id_at_location, self.path)
            # TODO: Think of a way to free the memory
            # del node.store[arg.id_at_location]
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
        value = resolved_self.data
        if isinstance(value, Tensor):
            nr_parties = value.child.child.nr_parties
            rank = value.child.child.rank
        else:
            nr_parties = value.nr_parties
            rank = value.rank

        seed_id_locations = resolved_kwargs.get("seed_id_locations", None)
        if seed_id_locations is None:
            raise ValueError(
                "Expected 'seed_id_locations' to be in the kwargs to generate id_at_location in a deterministic matter"
            )

        resolved_kwargs.pop("seed_id_locations")

        actions_generator = smpc_action_functions.get_action_generator_from_op(
            operation_str=method_name, nr_parties=nr_parties
        )
        args_id = [arg.id_at_location for arg in self.args]

        # TODO: For the moment we don't run any SMPC operation that provides any kwarg
        kwargs = {
            "seed_id_locations": int(seed_id_locations),
            "node": node,
        }

        # Get the list of actions to be run
        # TODO : Remove client as we do not use it now.
        actions: Union[List[SMPCActionMessage], SMPCActionSeqBatchMessage]
        actions = actions_generator(*args_id, **kwargs)  # type: ignore

        if isinstance(actions, (list, tuple)) and isinstance(
            actions[0], SMPCActionMessage
        ):
            actions = SMPCActionMessage.filter_actions_after_rank(rank, actions)
            for action in actions:
                RunClassMethodSMPCAction.execute_smpc_action(node, action, verify_key)
        elif isinstance(actions, SMPCActionSeqBatchMessage):
            msg = actions
            while msg.smpc_actions:
                action = msg.smpc_actions[0]
                RunClassMethodSMPCAction.execute_smpc_action(node, action, verify_key)
                del msg.smpc_actions[0]

    @staticmethod
    def execute_smpc_action(
        node: AbstractNode, msg: "SMPCActionMessage", verify_key: VerifyKey
    ) -> None:
        # relative
        from .smpc_action_functions import _MAP_ACTION_TO_FUNCTION

        func = _MAP_ACTION_TO_FUNCTION[msg.name_action]
        store_object_self = node.store.get(key=msg.self_id)
        if store_object_self is None:
            raise KeyError("Object not already in store")

        _self = store_object_self.data
        args = [node.store.get(arg_id).data for arg_id in msg.args_id]

        kwargs = {}  # type: ignore
        for key, kwarg_id in msg.kwargs_id.items():
            data = node.store.get(kwarg_id).data
            if data is None:
                raise KeyError(f"Key {key} is not available")

            kwargs[key] = data
        kwargs = {**kwargs, **msg.kwargs}
        (
            upcasted_args,
            upcasted_kwargs,
        ) = lib.python.util.upcast_args_and_kwargs(args, kwargs)
        logger.warning(func)

        if msg.name_action in {"spdz_multiply", "spdz_mask"}:
            result = func(_self, *upcasted_args, **upcasted_kwargs, node=node)
        elif msg.name_action == "local_decomposition":
            result = func(
                _self,
                *upcasted_args,
                **upcasted_kwargs,
                node=node,
                read_permissions=store_object_self.read_permissions,
            )
        else:
            result = func(_self, *upcasted_args, **upcasted_kwargs)

        if lib.python.primitive_factory.isprimitive(value=result):
            # Wrap in a SyPrimitive
            result = lib.python.primitive_factory.PrimitiveFactory.generate_primitive(
                value=result, id=msg.id_at_location
            )
        else:
            # TODO: overload all methods to incorporate this automatically
            if hasattr(result, "id"):
                try:
                    if hasattr(result, "_id"):
                        # set the underlying id
                        result._id = msg.id_at_location
                    else:
                        result.id = msg.id_at_location

                    if result.id != msg.id_at_location:
                        raise AttributeError("IDs don't match")
                except AttributeError as e:
                    err = f"Unable to set id on result {type(result)}. {e}"
                    traceback_and_raise(Exception(err))

        if not isinstance(result, StorableObject):
            result = StorableObject(
                id=msg.id_at_location,
                data=result,
                read_permissions=store_object_self.read_permissions,
            )

        node.store[msg.id_at_location] = result

    def _object2proto(self) -> RunClassMethodSMPCAction_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: RunClassMethodSMPCAction_PB

        .. note::
            This method is purely an internal method. Please use sy.serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """

        return RunClassMethodSMPCAction_PB(
            path=self.path,
            _self=sy.serialize(self._self),
            args=list(map(lambda x: sy.serialize(x), self.args)),
            kwargs={k: sy.serialize(v) for k, v in self.kwargs.items()},
            id_at_location=sy.serialize(self.id_at_location),
            address=sy.serialize(self.address),
            msg_id=sy.serialize(self.id),
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
            _self=sy.deserialize(blob=proto._self),
            args=list(map(lambda x: sy.deserialize(blob=x), proto.args)),
            kwargs={k: sy.deserialize(blob=v) for k, v in proto.kwargs.items()},
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

        return RunClassMethodSMPCAction_PB
