# stdlib
import operator
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
import numpy as np

# syft absolute
from syft import lib

# syft relative
from .....proto.core.node.smpc.action.smpc_action_pb2 import SMPCAction as SMPCAction_PB
from ....common.serde.deserialize import _deserialize
from ....common.serde.serializable import bind_protobuf
from ....common.uid import UID
from ....io.address import Address
from ...common.action.common import ImmediateActionWithoutReply


def smpc_add(self_id, other_id, seed, node):
    # There is no need to use the generator for addition

    generator_id = np.random.default_rng(seed)
    other = node.store[other_id].data

    # TODO: Instaciate Actual Action with ImmediateActionWithoutReply
    actions = []
    if isinstance(other, ShareTensor):
        # All parties should add the other share if empty list
        actions.append(
            SMPCAction(
                "mpc_add",
                args_ids=[self_id, other_id],
                kwargs_ids={},
                ranks_to_run_action=[],
            )
        )
    else:
        # Only rank 0 (the first party) would add that public value
        actions.append(
            SMPCAction(
                "mpc_add",
                args_ids=[self_id, other_id],
                kwargs_ids={},
                ranks_to_run_action=[0],
            )
        )

    return actions


MAP_FUNC_TO_ACTION = {"__add__": smpc_add}


_MAP_ACTION_TO_FUNCTION = {
    "mpc_add": operator.add,
}


def SMPCExecute(actions, node):
    for action in actions:
        _try_action_with_retry(action, node)


@bind_protobuf
class SMPCAction(ImmediateActionWithoutReply):
    def __init__(
        name_action,
        self_id,
        args_id,
        kwargs_id,
        result_id,
        ranks_to_run_action,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        self.name_action = name_action
        self.self_id = self_id
        self.args_id = id_args
        self.kwargs_id = id_kwargs
        self.result_id = result_id
        self.ranks_to_run_action = ranks_to_run_action
        self.address = address
        self.msg_id = msg_id

    @staticmethod
    def filter_actions_after_rank(rank, actions):
        return [
            action
            for action in actions
            if x.rank_to_run_action == [] or rank in x.rank_to_run_action
        ]

    def execute_action(self, node: AbstractNode, verify_key: VerifyKey) -> None:
        operation, args_ids, kwargs_ids = action
        func = _MAP_ACTION_TO_FUNCTION[operation]

        args = None
        kwargs = None

        for i in range(10):
            try:
                args = [node.store[arg_id].data for arg_id in args_ids]
                kwargs = {
                    key: node.store[kwarg_id].data for key, kwarg_id in kwargs_ids
                }
                (
                    upcasted_args,
                    upcasted_kwargs,
                ) = lib.python.util.upcast_args_and_kwargs(args, kwargs)

                res = func(*upcasted_args, **upcasted_kwargs)
                break

            except KeyError:
                # For the object to reach the store and retry
                time.sleep(1)

        if args is None or kwargs is None:
            raise Exception("Abort since could not retrieve args/kwargs!")

        return res

    def _object2proto(self) -> SMPCAction_PB:
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

        return SMPCAction_PB(
            name_action=self.name_action,
            self_id=serialize(self.self_id),
            args_id=list(map(lambda x: serialize(x), self.args_id)),
            kwargs_id={k: serialize(v) for k, v in self.kwargs_id.items()},
            id_at_location=serialize(self.id_at_location),
            address=serialize(self.address),
            msg_id=serialize(self.id),
        )

    @staticmethod
    def _proto2object(proto: SMPCAction_PB) -> "SMPCAction":
        """Creates a ObjectWithID from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of RunClassMethodAction
        :rtype: RunClassMethodAction

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return SMPCAction(
            name_action=name_action,
            args_id=_deserialize(proto.args_id),
            kwargs_id={k: v for k, v in proto.kwargs_id.items()},
            id_at_location=_deserialize(blob=proto.id_at_location),
            address=_deserialize(blob=proto.address),
            msg_id=_deserialize(blob=proto.msg_id),
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

        return SMPCAction_PB
