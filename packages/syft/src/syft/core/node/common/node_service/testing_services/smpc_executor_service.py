# stdlib
from typing import List
from typing import Type

# third party
from nacl.signing import VerifyKey

# relative
from ...... import lib
from .....store.storeable_object import StorableObject
from ....abstract.node import AbstractNode
from ....common.action.smpc_action_message import SMPCActionMessage
from ....common.action.smpc_action_message import _MAP_ACTION_TO_FUNCTION
from ....common.node_service.node_service import ImmediateNodeServiceWithoutReply


class SMPCExecutorService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    def message_handler_types() -> List[Type[SMPCActionMessage]]:
        return [SMPCActionMessage]

    @staticmethod
    def process(
        node: AbstractNode, msg: SMPCActionMessage, verify_key: VerifyKey
    ) -> None:
        func = _MAP_ACTION_TO_FUNCTION[msg.name_action]
        store_object_self = node.store.get_object(key=msg.self_id)
        if store_object_self is None:
            raise KeyError("Object not already in store")

        _self = store_object_self.data
        args = [node.store[arg_id].data for arg_id in msg.args_id]
        kwargs = {}
        for key, kwarg_id in msg.kwargs_id.items():
            data = node.store[kwarg_id].data
            if data is None:
                raise KeyError(f"Key {key} is not available")

            kwargs[key] = data
        (
            upcasted_args,
            upcasted_kwargs,
        ) = lib.python.util.upcast_args_and_kwargs(args, kwargs)
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
