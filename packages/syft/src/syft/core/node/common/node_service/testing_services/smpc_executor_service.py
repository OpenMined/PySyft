# stdlib
import time
from typing import List
from typing import Optional
from typing import Type

# third party
from nacl.signing import VerifyKey

# relative
from ...... import lib
from ...... import logger
from ......logger import traceback_and_raise
from .....store.storeable_object import StorableObject
from ....abstract.node import AbstractNode
from ...action.smpc_action_message import SMPCActionMessage
from ...action.smpc_action_message import _MAP_ACTION_TO_FUNCTION
from ..node_service import ImmediateNodeServiceWithoutReply


class SMPCExecutorService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    def message_handler_types() -> List[Type[SMPCActionMessage]]:
        return [SMPCActionMessage]

    @staticmethod
    def process(
        node: AbstractNode,
        msg: SMPCActionMessage,
        verify_key: Optional[VerifyKey] = None,
    ) -> None:
        """Given an SMPCAction, execute it (this action is sent to the node
        by the RabitMQ task)

        Attributes:
            node (AbstractNode): the node that received the message
            msg (SMPCActionMessage): the message that should be executed
            verify_key (VerifyKey): the verify_key
        """
        func = _MAP_ACTION_TO_FUNCTION[msg.name_action]
        try:
            # keep five minutes seconds as max latency
            # TODO: Before Merge Should discuss with syft core team on max time.
            ctr = 3000
            while True:
                store_object_self = node.store.get_object(key=msg.self_id)
                if store_object_self is None:
                    ctr -= 1
                    time.sleep(0.1)
                    # We intimate user every ten seconds
                    if ctr % 100 == 0:
                        print("Waiting for Object to arrive...🚀")
                else:
                    break
                if ctr <= 0:
                    raise Exception(
                        "Object not found or Object did not arrive at store"
                    )
        except Exception as e:
            log = (
                f"Unable to Get Object with ID {msg.self_id} from store. "
                + f"Possible dangling Pointer. {e}"
            )

            traceback_and_raise(Exception(log))

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
        logger.warning(func)
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
