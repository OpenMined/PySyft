import syft as sy
from syft.workers.abstract import AbstractWorker

# from syft.frameworks.torch.tensors.interpreters.placeholder import PlaceHolder

# from syft_proto.execution.v1.operation_pb2 import Operation as OperationPB
# from syft_proto.types.syft.v1.arg_pb2 import Arg as ArgPB


class Operation:
    """All syft operations use this message type

    In Syft, an operation is when one worker wishes to tell another worker to do something with
    objects contained in the worker._objects registry (or whatever the official object store is
    backed with in the case that it's been overridden). Semantically, one could view all Messages
    as a kind of operation, but when we say operation this is what we mean. For example, telling a
    worker to take two tensors and add them together is an operation. However, sending an object
    from one worker to another is not an operation (and would instead use the ObjectMessage type)."""

    def __init__(self, cmd_name, cmd_owner, cmd_args, cmd_kwargs, return_ids):
        """Initialize an operation message

        Args:
            message (Tuple): this is typically the args and kwargs of a method call on the client, but it
                can be any information necessary to execute the operation properly.
            return_ids (Tuple): primarily for our async infrastructure (Plan, Protocol, etc.), the id of
                operation results are set by the client. This allows the client to be able to predict where
                the results will be ahead of time. Importantly, this allows the client to pre-initalize the
                pointers to the future data, regardless of whether the operation has yet executed. It also
                reduces the size of the response from the operation (which is very often empty).

        """

        # call the parent constructor - setting the type integer correctly
        super().__init__()

        self.cmd_name = cmd_name
        self.cmd_owner = cmd_owner
        self.cmd_args = cmd_args
        self.cmd_kwargs = cmd_kwargs
        self.return_ids = return_ids

    @property
    def contents(self):
        """Return a tuple with the contents of the operation (backwards compatability)

        Some of our codebase still assumes that all message types have a .contents attribute. However,
        the contents attribute is very opaque in that it doesn't put any constraints on what the contents
        might be. Since we know this message is a operation, we instead choose to store contents in two pieces,
        self.message and self.return_ids, which allows for more efficient simplification (we don't have to
        simplify return_ids because they are always a list of integers, meaning they're already simplified)."""

        message = (self.cmd_name, self.cmd_owner, self.cmd_args, self.cmd_kwargs)

        return (message, self.return_ids)

    @staticmethod
    def simplify(worker: AbstractWorker, ptr: "Operation") -> tuple:
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
        message = (ptr.cmd_name, ptr.cmd_owner, ptr.cmd_args, ptr.cmd_kwargs)

        return (
            sy.serde.msgpack.serde._simplify(worker, message),
            sy.serde.msgpack.serde._simplify(worker, ptr.return_ids),
        )

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) -> "Operation":
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

        cmd_name = detailed_msg[0]
        cmd_owner = detailed_msg[1]
        cmd_args = detailed_msg[2]
        cmd_kwargs = detailed_msg[3]

        return Operation(cmd_name, cmd_owner, cmd_args, cmd_kwargs, detailed_ids)

    # @staticmethod
    # def bufferize(worker: AbstractWorker, operation: "Operation") -> "OperationPB":
    #     """
    #     This function takes the attributes of a OperationMessage and saves them in Protobuf
    #     Args:
    #         worker (AbstractWorker): a reference to the worker doing the serialization
    #         ptr (OperationMessage): an OperationMessage
    #     Returns:
    #         protobuf_obj: a Protobuf message holding the unique attributes of the message
    #     Examples:
    #         data = bufferize(message)
    #     """
    #     protobuf_op_msg = OperationMessagePB()
    #     protobuf_op = OperationPB()
    #     protobuf_op.command = operation.cmd_name

    #     if type(operation.cmd_owner) == sy.generic.pointers.pointer_tensor.PointerTensor:
    #         protobuf_owner = protobuf_op.owner_pointer
    #     elif (
    #         type(operation.cmd_owner)
    #         == sy.frameworks.torch.tensors.interpreters.placeholder.PlaceHolder
    #     ):
    #         protobuf_owner = protobuf_op.owner_placeholder
    #     else:
    #         protobuf_owner = protobuf_op.owner_tensor

    #     if operation.cmd_owner is not None:
    #         protobuf_owner.CopyFrom(sy.serde.protobuf.serde._bufferize(worker, operation.cmd_owner))

    #     if operation.cmd_args:
    #         protobuf_op.args.extend(OperationMessage._bufferize_args(worker, operation.cmd_args))

    #     if operation.cmd_kwargs:
    #         for key, value in operation.cmd_kwargs.items():
    #             protobuf_op.kwargs.get_or_create(key).CopyFrom(
    #                 OperationMessage._bufferize_arg(worker, value)
    #             )

    #     if operation.return_ids is not None:
    #         if type(operation.return_ids) == PlaceHolder:
    #             return_ids = list((operation.return_ids,))
    #         else:
    #             return_ids = operation.return_ids

    #         for return_id in return_ids:
    #             if type(return_id) == PlaceHolder:
    #                 protobuf_op.return_placeholders.extend(
    #                     [sy.serde.protobuf.serde._bufferize(worker, return_id)]
    #                 )
    #             else:
    #                 sy.serde.protobuf.proto.set_protobuf_id(protobuf_op.return_ids.add(), return_id)

    #     protobuf_op_msg.operation.CopyFrom(protobuf_op)
    #     return protobuf_op_msg

    # @staticmethod
    # def unbufferize(
    #     worker: AbstractWorker, protobuf_obj: "OperationMessagePB"
    # ) -> "OperationMessage":
    #     """
    #     This function takes the Protobuf version of this message and converts
    #     it into an OperationMessage. The bufferize() method runs the inverse of this method.

    #     Args:
    #         worker (AbstractWorker): a reference to the worker necessary for detailing. Read
    #             syft/serde/serde.py for more information on why this is necessary.
    #         protobuf_obj (OperationMessagePB): the Protobuf message

    #     Returns:
    #         obj (OperationMessage): an OperationMessage

    #     Examples:
    #         message = unbufferize(sy.local_worker, protobuf_msg)
    #     """

    #     command = protobuf_obj.operation.command
    #     protobuf_owner = protobuf_obj.operation.WhichOneof("owner")
    #     if protobuf_owner:
    #         owner = sy.serde.protobuf.serde._unbufferize(
    #             worker, getattr(protobuf_obj.operation, protobuf_obj.operation.WhichOneof("owner"))
    #         )
    #     else:
    #         owner = None
    #     args = OperationMessage._unbufferize_args(worker, protobuf_obj.operation.args)

    #     kwargs = {}
    #     for key in protobuf_obj.operation.kwargs:
    #         kwargs[key] = OperationMessage._unbufferize_arg(
    #             worker, protobuf_obj.operation.kwargs[key]
    #         )

    #     return_ids = [
    #         sy.serde.protobuf.proto.get_protobuf_id(pb_id)
    #         for pb_id in protobuf_obj.operation.return_ids
    #     ]

    #     return_placeholders = [
    #         sy.serde.protobuf.serde._unbufferize(worker, placeholder)
    #         for placeholder in protobuf_obj.operation.return_placeholders
    #     ]

    #     if return_placeholders:
    #         if len(return_placeholders) == 1:
    #             operation_msg = OperationMessage(
    #                 command, owner, tuple(args), kwargs, return_placeholders[0]
    #             )
    #         else:
    #             operation_msg = OperationMessage(
    #                 command, owner, tuple(args), kwargs, return_placeholders
    #             )
    #     else:
    #         operation_msg = OperationMessage(command, owner, tuple(args), kwargs, tuple(return_ids))

    #     return operation_msg

    # @staticmethod
    # def _bufferize_args(worker: AbstractWorker, args: list) -> list:
    #     protobuf_args = []
    #     for arg in args:
    #         protobuf_args.append(OperationMessage._bufferize_arg(worker, arg))
    #     return protobuf_args

    # @staticmethod
    # def _bufferize_arg(worker: AbstractWorker, arg: object) -> ArgPB:
    #     protobuf_arg = ArgPB()
    #     try:
    #         setattr(protobuf_arg, "arg_" + type(arg).__name__.lower(), arg)
    #     except:
    #         getattr(protobuf_arg, "arg_" + type(arg).__name__.lower()).CopyFrom(
    #             sy.serde.protobuf.serde._bufferize(worker, arg)
    #         )
    #     return protobuf_arg

    # @staticmethod
    # def _unbufferize_args(worker: AbstractWorker, protobuf_args: list) -> list:
    #     args = []
    #     for protobuf_arg in protobuf_args:
    #         args.append(OperationMessage._unbufferize_arg(worker, protobuf_arg))
    #     return args

    # @staticmethod
    # def _unbufferize_arg(worker: AbstractWorker, protobuf_arg: ArgPB) -> object:
    #     protobuf_arg_field = getattr(protobuf_arg, protobuf_arg.WhichOneof("arg"))
    #     try:
    #         arg = sy.serde.protobuf.serde._unbufferize(worker, protobuf_arg_field)
    #     except:
    #         arg = protobuf_arg_field
    #     return arg
