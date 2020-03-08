from typing import List
from typing import Union

import syft as sy
from syft.workers.abstract import AbstractWorker

from syft.execution.action import Action

from syft_proto.execution.v1.communication_action_pb2 import (
    CommunicationAction as CommunicationActionPB,
)
from syft_proto.types.syft.v1.arg_pb2 import Arg as ArgPB


class CommunicationAction(Action):
    """Describes communication actions performed on tensors"""

    def __init__(self, obj, source: Union[str, int], destinations: List[Union[str, int]], kwargs):
        """Initialize an communication action

        Args:
        """
        super().__init__()

        self.obj = obj
        self.source = source
        self.destinations = destinations
        self.kwargs = kwargs

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
            sy.serde.msgpack.serde._simplify(worker, communication.obj),
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

        (obj, source, destinations, kwargs) = communication_tuple

        detailed_obj = sy.serde.msgpack.serde._detail(worker, obj)
        detailed_source = sy.serde.msgpack.serde._detail(worker, source)
        detailed_destinations = sy.serde.msgpack.serde._detail(worker, destinations)
        detailed_kwargs = sy.serde.msgpack.serde._detail(worker, kwargs)

        return CommunicationAction(
            detailed_obj, detailed_source, detailed_destinations, detailed_kwargs
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

        protobuf_obj.obj = sy.serde.protobuf.serde._bufferize(worker, communication.obj)

        protobuf_obj.source = communication.source
        protobuf_obj.destinations = communication.destinations

        if operation.kwargs:
            for key, value in operation.kwargs.items():
                protobuf_op.kwargs.get_or_create(key).CopyFrom(
                    CommunicationAction._bufferize_arg(worker, value)
                )

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
            obj (CommunicationAction): a CommunicationAction

        Examples:
            message = unbufferize(sy.local_worker, protobuf_msg)
        """
        obj = sy.serde.protobuf.serde._unbufferize(worker, protobuf_obj.obj)

        source = protobuf_obj.source
        destinations = protobuf_obj.destinations

        kwargs = {
            key: ComputationAction._unbufferize_arg(worker, kwarg)
            for key, kwarg in protobuf_obj.items()
        }

        return CommunicationAction(obj, source, destinations, **kwargs)

    @staticmethod
    def _bufferize_args(worker: AbstractWorker, args: list) -> list:
        protobuf_args = []
        for arg in args:
            protobuf_args.append(ComputationAction._bufferize_arg(worker, arg))
        return protobuf_args

    @staticmethod
    def _bufferize_arg(worker: AbstractWorker, arg: object) -> ArgPB:
        protobuf_arg = ArgPB()
        try:
            setattr(protobuf_arg, "arg_" + type(arg).__name__.lower(), arg)
        except:
            getattr(protobuf_arg, "arg_" + type(arg).__name__.lower()).CopyFrom(
                sy.serde.protobuf.serde._bufferize(worker, arg)
            )
        return protobuf_arg

    @staticmethod
    def _unbufferize_args(worker: AbstractWorker, protobuf_args: list) -> list:
        args = []
        for protobuf_arg in protobuf_args:
            args.append(ComputationAction._unbufferize_arg(worker, protobuf_arg))
        return args

    @staticmethod
    def _unbufferize_arg(worker: AbstractWorker, protobuf_arg: ArgPB) -> object:
        protobuf_arg_field = getattr(protobuf_arg, protobuf_arg.WhichOneof("arg"))
        try:
            arg = sy.serde.protobuf.serde._unbufferize(worker, protobuf_arg_field)
        except:
            arg = protobuf_arg_field
        return arg
