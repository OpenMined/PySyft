from typing import List, Union, Tuple

import syft as sy
from syft.execution.action import Action
from syft.workers.abstract import AbstractWorker

from syft_proto.execution.v1.communication_action_pb2 import (
    CommunicationAction as CommunicationActionPB,
)


COMMUNICATION_METHODS = ["move", "remote_send", "mid_get", "remote_get", "get", "share", "share_"]


class CommunicationAction(Action):
    """Describes communication actions performed on tensors"""

    def __init__(self, name, target, args_, kwargs_, return_ids, return_value=False):
        """Initialize an action

        Args:
            name (String): The name of the method to be invoked (e.g. "send")
            target (Tensor): The object to invoke the method on
            args_ (Tuple): The arguments to the method call
            kwargs_ (Dictionary): The keyword arguments to the method call
            return_ids (Tuple): primarily for our async infrastructure (Plan, Protocol, etc.), the id of
                action results are set by the client. This allows the client to be able to predict where
                the results will be ahead of time. Importantly, this allows the client to pre-initalize the
                pointers to the future data, regardless of whether the action has yet executed. It also
                reduces the size of the response from the action (which is very often empty).
            return_value (boolean): return the result or not. If true, the result is directly returned,
                if not, the command sender will create a pointer to the remote result using the return_ids
                and will need to do .get() later to get the result.

        """
        if name not in COMMUNICATION_METHODS:
            raise ValueError(
                f"Method `{name}` is not supported by CommunicationActions. Consider using ComputationAction instead."
            )

        super().__init__(name, target, args_, kwargs_, return_ids, return_value=return_value)

    @staticmethod
    def simplify(worker: AbstractWorker, action: "Action") -> tuple:
        """
        This function takes the attributes of a CommunicationAction and saves them in a tuple
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            action (CommunicationAction): a CommunicationAction
        Returns:
            tuple: a tuple holding the unique attributes of the CommunicationAction
        Examples:
            data = simplify(worker, action)
        """
        return Action.simplify(worker, action)

    @staticmethod
    def detail(worker: AbstractWorker, action_tuple: tuple) -> "Action":
        """
        This function takes the simplified tuple version of this message and converts
        it into a CommunicationAction. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            communication_tuple (Tuple): the raw information being detailed.
        Returns:
            communication (CommunicationAction): a CommunicationAction.
        Examples:
            communication = detail(sy.local_worker, communication_tuple)
        """
        attrs = Action.detail(worker, action_tuple)

        return CommunicationAction(*attrs)

    @staticmethod
    def bufferize(
        worker: AbstractWorker, communication: "CommunicationAction"
    ) -> "CommunicationActionPB":
        """
        This function takes the attributes of a CommunicationAction and saves them in Protobuf
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            communication (CommunicationAction): a CommunicationAction
        Returns:
            protobuf_obj: a Protobuf message holding the unique attributes of the communication
        Examples:
            data = bufferize(sy.local_worker, communication)
        """
        protobuf_obj = CommunicationActionPB()
        protobuf_obj.name = communication.name

        sy.serde.protobuf.proto.set_protobuf_id(protobuf_obj.obj_id, communication.obj_id)
        sy.serde.protobuf.proto.set_protobuf_id(protobuf_obj.source, communication.source)

        for destination in communication.destinations:
            sy.serde.protobuf.proto.set_protobuf_id(protobuf_obj.destinations.add(), destination)

        if communication.kwargs:
            for key, value in communication.kwargs.items():
                protobuf_obj.kwargs.get_or_create(key).CopyFrom(
                    sy.serde.protobuf.serde.bufferize_arg(worker, value)
                )

        return protobuf_obj

    @staticmethod
    def unbufferize(
        worker: AbstractWorker, protobuf_obj: "CommunicationActionPB"
    ) -> "CommunicationAction":
        """
        This function takes the Protobuf version of this message and converts
        it into a CommunicationAction. The bufferize() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            protobuf_obj (CommunicationActionPB): the Protobuf message

        Returns:
            obj_id (CommunicationAction): a CommunicationAction

        Examples:
            message = unbufferize(sy.local_worker, protobuf_msg)
        """
        name = protobuf_obj.name

        obj_id = sy.serde.protobuf.proto.get_protobuf_id(protobuf_obj.obj_id)
        source = sy.serde.protobuf.proto.get_protobuf_id(protobuf_obj.source)
        destinations = [
            sy.serde.protobuf.proto.get_protobuf_id(pb_id) for pb_id in protobuf_obj.destinations
        ]

        kwargs_ = {
            key: sy.serde.protobuf.serde.unbufferize_arg(worker, kwarg)
            for key, kwarg in protobuf_obj.kwargs.items()
        }

        return CommunicationAction(obj_id, name, source, destinations, kwargs_)
