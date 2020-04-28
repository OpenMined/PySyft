from typing import List, Union, Tuple

import syft as sy
from syft.execution.action import Action
from syft.workers.abstract import AbstractWorker

from syft_proto.execution.v1.communication_action_pb2 import (
    CommunicationAction as CommunicationActionPB,
)


class CommunicationAction(Action):
    """Describes communication actions performed on tensors"""

    def __init__(
        self,
        obj_id: Union[str, int],  # id of target object (almost always) pointer_tensor
        name: str,  # the pointer_tensor method # @TODO: reactor to enum
        source: Union[str, int],  # id of the sender (worker)
        destinations: List[Union[str, int]],  # id of the receiver(s) (worker(s))
        kwargs_: dict,  # key word args needed for the pointer tensor method == self.name
    ):
        """Initialize an communication action

        Args:
        """
        super().__init__()

        self.obj_id = obj_id
        if name in ["move", "remote_send", "mid_get", "remote_get", "get", "share", "share_"]:
            #  float_prec, fix_prec => should be computation actions (they modify tensors)?
            self.name = name
        else:
            raise ValueError(
                f"name `{name}` for CommunicationAction is not in the list of supported actions"
            )
        self.source = source
        self.destinations = destinations
        self.kwargs = kwargs_

    @property
    def contents(self):
        """Return a tuple with the contents of the operation (backwards compatability)."""

        return (self.obj_id, self.name, self.source, self.destinations, self.kwargs)

    def __eq__(self, other):
        return (
            self.obj_id == other.obj_id
            and self.name == other.name
            and self.source == other.source
            and self.destinations == other.destinations
            and self.kwargs == other.kwargs
        )

    @staticmethod
    def simplify(worker: AbstractWorker, communication: "CommunicationAction") -> tuple:
        """
        This function takes the attributes of a CommunicationAction and saves them in a tuple
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            communication (CommunicationAction): a CommunicationAction
        Returns:
            tuple: a tuple holding the unique attributes of the CommunicationAction
        Examples:
            data = simplify(worker, communication)
        """
        return (
            sy.serde.msgpack.serde._simplify(worker, communication.obj_id),
            sy.serde.msgpack.serde._simplify(worker, communication.name),
            sy.serde.msgpack.serde._simplify(worker, communication.source),
            sy.serde.msgpack.serde._simplify(worker, communication.destinations),
            sy.serde.msgpack.serde._simplify(worker, communication.kwargs),
        )

    @staticmethod
    def detail(worker: AbstractWorker, communication_tuple: tuple) -> "CommunicationAction":
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

        (obj_id, name, source, destinations, kwargs_) = communication_tuple

        detailed_obj = sy.serde.msgpack.serde._detail(worker, obj_id)
        detailed_name = sy.serde.msgpack.serde._detail(worker, name)
        detailed_source = sy.serde.msgpack.serde._detail(worker, source)
        detailed_destinations = sy.serde.msgpack.serde._detail(worker, destinations)
        detailed_kwargs = sy.serde.msgpack.serde._detail(worker, kwargs_)

        return CommunicationAction(
            detailed_obj, detailed_name, detailed_source, detailed_destinations, detailed_kwargs
        )

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
