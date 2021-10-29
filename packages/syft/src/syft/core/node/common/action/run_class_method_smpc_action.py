# stdlib
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# syft absolute
import syft as sy

# relative
from ..... import lib
from .....logger import critical
from .....proto.core.node.common.action.run_class_method_smpc_pb2 import (
    RunClassMethodSMPCAction as RunClassMethodSMPCAction_PB,
)
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ....io.address import Address
from ...abstract.node import AbstractNode
from .common import ImmediateActionWithoutReply
from .exceptions import ObjectNotInStore


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
        from ..... import Tensor
        from .smpc_action_message import SMPCActionMessage

        resolved_self = node.store.get_object(key=self._self.id_at_location)

        if resolved_self is None:
            critical(
                f"execute_action on {self.path} failed due to missing object"
                + f" at: {self._self.id_at_location}"
            )
            raise ObjectNotInStore
        result_read_permissions = resolved_self.read_permissions

        resolved_args = list()
        tag_args = []
        for arg in self.args:
            r_arg = node.store.get_object(key=arg.id_at_location)
            if r_arg is None:
                critical(
                    f"execute_action on {self.path} failed due to missing object"
                    + f" at: {arg.id_at_location}"
                )
                raise ObjectNotInStore

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
            r_arg = node.store.get_object(arg.id_at_location)
            if r_arg is None:
                critical(
                    f"execute_action on {self.path} failed due to missing object"
                    + f" at: {arg.id_at_location}"
                )
                raise ObjectNotInStore
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

        client = resolved_kwargs.get("client", None)
        if client is None:
            raise ValueError(
                "Expected client to be in the kwargs to generate SMPCActionMessage"
            )

        resolved_kwargs.pop("client")
        actions_generator = SMPCActionMessage.get_action_generator_from_op(
            operation_str=method_name, nr_parties=nr_parties
        )
        args_id = [arg.id_at_location for arg in self.args]

        # TODO: For the moment we don't run any SMPC operation that provides any kwarg
        kwargs = {
            "seed_id_locations": int(seed_id_locations),
            "node": node,
            "client": client,
        }

        # Get the list of actions to be run
        actions = actions_generator(*args_id, **kwargs)  # type: ignore
        actions = SMPCActionMessage.filter_actions_after_rank(rank, actions)
        base_url = client.routes[0].connection.base_url
        client.routes[0].connection.base_url = base_url.replace(
            "localhost", "docker-host"
        )
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
