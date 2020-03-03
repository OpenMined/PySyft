import syft as sy
from syft.workers.abstract import AbstractWorker

from syft.execution.action import Action
from syft.frameworks.torch.tensors.interpreters.placeholder import PlaceHolder

from syft_proto.execution.v1.operation_pb2 import Operation as OperationPB
from syft_proto.types.syft.v1.arg_pb2 import Arg as ArgPB


class ComputationAction(Action):
    """Describes mathematical operations performed on tensors"""

    def __init__(self, name, operand, args_, kwargs_, return_ids):
        """Initialize an operation

        Args:
            name (String): The name of the method to be invoked (e.g. "__add__")
            operand (Tensor): The object to invoke the method on
            args_ (Tuple): The arguments to the method call
            kwargs_ (Dictionary): The keyword arguments to the method call
            return_ids (Tuple): primarily for our async infrastructure (Plan, Protocol, etc.), the id of
                operation results are set by the client. This allows the client to be able to predict where
                the results will be ahead of time. Importantly, this allows the client to pre-initalize the
                pointers to the future data, regardless of whether the operation has yet executed. It also
                reduces the size of the response from the operation (which is very often empty).

        """

        # call the parent constructor - setting the type integer correctly
        super().__init__()

        self.name = name
        self.operand = operand
        self.args = args_
        self.kwargs = kwargs_
        self.return_ids = return_ids

    @property
    def contents(self):
        """Return a tuple with the contents of the operation (backwards compatability)

        Some of our codebase still assumes that all message types have a .contents attribute. However,
        the contents attribute is very opaque in that it doesn't put any constraints on what the contents
        might be. Since we know this message is a operation, we instead choose to store contents in two pieces,
        self.message and self.return_ids, which allows for more efficient simplification (we don't have to
        simplify return_ids because they are always a list of integers, meaning they're already simplified)."""

        message = (self.name, self.operand, self.args, self.kwargs)

        return (message, self.return_ids)

    @staticmethod
    def simplify(worker: AbstractWorker, ptr: "ComputationAction") -> tuple:
        """
        This function takes the attributes of a Operation and saves them in a tuple
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            ptr (Operation): a Message
        Returns:
            tuple: a tuple holding the unique attributes of the message
        Examples:
            data = simplify(ptr)
        """
        # NOTE: we can skip calling _simplify on return_ids because they should already be
        # a list of simple types.
        message = (ptr.name, ptr.operand, ptr.args, ptr.kwargs)

        return (
            sy.serde.msgpack.serde._simplify(worker, message),
            sy.serde.msgpack.serde._simplify(worker, ptr.return_ids),
        )

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) -> "ComputationAction":
        """
        This function takes the simplified tuple version of this message and converts
        it into a Operation. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.
        Returns:
            ptr (Operation): an Operation.
        Examples:
            message = detail(sy.local_worker, msg_tuple)
        """
        message = msg_tuple[0]
        return_ids = msg_tuple[1]

        detailed_msg = sy.serde.msgpack.serde._detail(worker, message)
        detailed_ids = sy.serde.msgpack.serde._detail(worker, return_ids)

        name, operand, args_, kwargs_ = detailed_msg

        return ComputationAction(name, operand, args_, kwargs_, detailed_ids)

    @staticmethod
    def bufferize(worker: AbstractWorker, operation: "ComputationAction") -> "OperationPB":
        """
        This function takes the attributes of a Operation and saves them in Protobuf
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            operation (Operation): an Operation
        Returns:
            protobuf_obj: a Protobuf message holding the unique attributes of the message
        Examples:
            data = bufferize(message)
        """
        protobuf_op = OperationPB()
        protobuf_op.command = operation.name

        if type(operation.operand) == sy.generic.pointers.pointer_tensor.PointerTensor:
            protobuf_owner = protobuf_op.owner_pointer
        elif (
            type(operation.operand)
            == sy.frameworks.torch.tensors.interpreters.placeholder.PlaceHolder
        ):
            protobuf_owner = protobuf_op.owner_placeholder
        else:
            protobuf_owner = protobuf_op.owner_tensor

        if operation.operand is not None:
            protobuf_owner.CopyFrom(sy.serde.protobuf.serde._bufferize(worker, operation.operand))

        if operation.args:
            protobuf_op.args.extend(ComputationAction._bufferize_args(worker, operation.args))

        if operation.kwargs:
            for key, value in operation.kwargs.items():
                protobuf_op.kwargs.get_or_create(key).CopyFrom(
                    ComputationAction._bufferize_arg(worker, value)
                )

        if operation.return_ids is not None:
            if type(operation.return_ids) == PlaceHolder:
                return_ids = list((operation.return_ids,))
            else:
                return_ids = operation.return_ids

            for return_id in return_ids:
                if type(return_id) == PlaceHolder:
                    protobuf_op.return_placeholders.extend(
                        [sy.serde.protobuf.serde._bufferize(worker, return_id)]
                    )
                else:
                    sy.serde.protobuf.proto.set_protobuf_id(protobuf_op.return_ids.add(), return_id)

        return protobuf_op

    @staticmethod
    def unbufferize(worker: AbstractWorker, protobuf_obj: "OperationPB") -> "ComputationAction":
        """
        This function takes the Protobuf version of this message and converts
        it into an Operation. The bufferize() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            protobuf_obj (OperationPB): the Protobuf message

        Returns:
            obj (Operation): an Operation

        Examples:
            message = unbufferize(sy.local_worker, protobuf_msg)
        """
        command = protobuf_obj.command
        protobuf_owner = protobuf_obj.WhichOneof("owner")
        if protobuf_owner:
            owner = sy.serde.protobuf.serde._unbufferize(
                worker, getattr(protobuf_obj, protobuf_obj.WhichOneof("owner"))
            )
        else:
            owner = None
        args = ComputationAction._unbufferize_args(worker, protobuf_obj.args)

        kwargs = {}
        for key in protobuf_obj.kwargs:
            kwargs[key] = ComputationAction._unbufferize_arg(worker, protobuf_obj.kwargs[key])

        return_ids = [
            sy.serde.protobuf.proto.get_protobuf_id(pb_id) for pb_id in protobuf_obj.return_ids
        ]

        return_placeholders = [
            sy.serde.protobuf.serde._unbufferize(worker, placeholder)
            for placeholder in protobuf_obj.return_placeholders
        ]

        if return_placeholders:
            if len(return_placeholders) == 1:
                operation = ComputationAction(
                    command, owner, tuple(args), kwargs, return_placeholders[0]
                )
            else:
                operation = ComputationAction(
                    command, owner, tuple(args), kwargs, return_placeholders
                )
        else:
            operation = ComputationAction(command, owner, tuple(args), kwargs, tuple(return_ids))

        return operation

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
