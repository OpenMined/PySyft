import syft as sy
from syft.workers.abstract import AbstractWorker

from syft.execution.action import Action
from syft.execution.placeholder import PlaceHolder

from syft_proto.execution.v1.computation_action_pb2 import ComputationAction as ComputationActionPB


class ComputationAction(Action):
    """Describes mathematical operations performed on tensors"""

    def __init__(self, name, target, args_, kwargs_, return_ids):
        """Initialize an action

        Args:
            name (String): The name of the method to be invoked (e.g. "__add__")
            target (Tensor): The object to invoke the method on
            args_ (Tuple): The arguments to the method call
            kwargs_ (Dictionary): The keyword arguments to the method call
            return_ids (Tuple): primarily for our async infrastructure (Plan, Protocol, etc.), the id of
                action results are set by the client. This allows the client to be able to predict where
                the results will be ahead of time. Importantly, this allows the client to pre-initalize the
                pointers to the future data, regardless of whether the action has yet executed. It also
                reduces the size of the response from the action (which is very often empty).

        """

        # call the parent constructor - setting the type integer correctly
        super().__init__()

        self.name = name
        self.target = target
        self.args = args_
        self.kwargs = kwargs_
        self.return_ids = return_ids

    @property
    def contents(self):
        """Return a tuple with the contents of the action (backwards compatability)

        Some of our codebase still assumes that all message types have a .contents attribute. However,
        the contents attribute is very opaque in that it doesn't put any constraints on what the contents
        might be. Since we know this message is a action, we instead choose to store contents in two pieces,
        self.message and self.return_ids, which allows for more efficient simplification (we don't have to
        simplify return_ids because they are always a list of integers, meaning they're already simplified)."""

        message = (self.name, self.target, self.args, self.kwargs)

        return (message, self.return_ids)

    @staticmethod
    def simplify(worker: AbstractWorker, action: "ComputationAction") -> tuple:
        """
        This function takes the attributes of a Action and saves them in a tuple
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            action (ComputationAction): the ComputationAction object to simplify
        Returns:
            tuple: a tuple holding the unique attributes of the message
        Examples:
            data = simplify(sy.local_worker, action)
        """
        # NOTE: we can skip calling _simplify on return_ids because they should already be
        # a list of simple types.
        message = (action.name, action.target, action.args, action.kwargs)

        return (
            sy.serde.msgpack.serde._simplify(worker, message),
            sy.serde.msgpack.serde._simplify(worker, action.return_ids),
        )

    @staticmethod
    def detail(worker: AbstractWorker, msg_tuple: tuple) -> "ComputationAction":
        """
        This function takes the simplified tuple version of this message and converts
        it into a Action. The simplify() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            msg_tuple (Tuple): the raw information being detailed.
        Returns:
            action (ComputationAction): a ComputationAction.
        Examples:
            action = detail(sy.local_worker, msg_tuple)
        """
        message = msg_tuple[0]
        return_ids = msg_tuple[1]

        detailed_msg = sy.serde.msgpack.serde._detail(worker, message)
        detailed_ids = sy.serde.msgpack.serde._detail(worker, return_ids)

        name, target, args_, kwargs_ = detailed_msg

        return ComputationAction(name, target, args_, kwargs_, detailed_ids)

    @staticmethod
    def bufferize(worker: AbstractWorker, action: "ComputationAction") -> "ComputationActionPB":
        """
        This function takes the attributes of a Action and saves them in Protobuf
        Args:
            worker (AbstractWorker): a reference to the worker doing the serialization
            action (Action): an Action
        Returns:
            protobuf_obj: a Protobuf message holding the unique attributes of the message
        Examples:
            data = bufferize(message)
        """
        protobuf_op = ComputationActionPB()
        protobuf_op.command = action.name

        if type(action.target) == sy.generic.pointers.pointer_tensor.PointerTensor:
            protobuf_owner = protobuf_op.target_pointer
        elif type(action.target) == sy.execution.placeholder.PlaceHolder:
            protobuf_owner = protobuf_op.target_placeholder
        else:
            protobuf_owner = protobuf_op.target_tensor

        if action.target is not None:
            protobuf_owner.CopyFrom(sy.serde.protobuf.serde._bufferize(worker, action.target))

        if action.args:
            protobuf_op.args.extend(sy.serde.protobuf.serde.bufferize_args(worker, action.args))

        if action.kwargs:
            for key, value in action.kwargs.items():
                protobuf_op.kwargs.get_or_create(key).CopyFrom(
                    sy.serde.protobuf.serde.bufferize_arg(worker, value)
                )

        if action.return_ids is not None:
            if type(action.return_ids) == PlaceHolder:
                return_ids = list((action.return_ids,))
            else:
                return_ids = action.return_ids

            for return_id in return_ids:
                if type(return_id) == PlaceHolder:
                    protobuf_op.return_placeholders.extend(
                        [sy.serde.protobuf.serde._bufferize(worker, return_id)]
                    )
                else:
                    sy.serde.protobuf.proto.set_protobuf_id(protobuf_op.return_ids.add(), return_id)

        return protobuf_op

    @staticmethod
    def unbufferize(
        worker: AbstractWorker, protobuf_obj: "ComputationActionPB"
    ) -> "ComputationAction":
        """
        This function takes the Protobuf version of this message and converts
        it into an Action. The bufferize() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            protobuf_obj (ComputationActionPB): the Protobuf message

        Returns:
            obj (ComputationAction): a ComputationAction

        Examples:
            message = unbufferize(sy.local_worker, protobuf_msg)
        """
        command = protobuf_obj.command
        protobuf_target = protobuf_obj.WhichOneof("target")
        if protobuf_target:
            target = sy.serde.protobuf.serde._unbufferize(
                worker, getattr(protobuf_obj, protobuf_obj.WhichOneof("target"))
            )
        else:
            target = None
        args = sy.serde.protobuf.serde.unbufferize_args(worker, protobuf_obj.args)

        kwargs = {}
        for key in protobuf_obj.kwargs:
            kwargs[key] = sy.serde.protobuf.serde.unbufferize_arg(worker, protobuf_obj.kwargs[key])

        return_ids = [
            sy.serde.protobuf.proto.get_protobuf_id(pb_id) for pb_id in protobuf_obj.return_ids
        ]

        return_placeholders = [
            sy.serde.protobuf.serde._unbufferize(worker, placeholder)
            for placeholder in protobuf_obj.return_placeholders
        ]

        if return_placeholders:
            if len(return_placeholders) == 1:
                action = ComputationAction(
                    command, target, tuple(args), kwargs, return_placeholders[0]
                )
            else:
                action = ComputationAction(
                    command, target, tuple(args), kwargs, return_placeholders
                )
        else:
            action = ComputationAction(command, target, tuple(args), kwargs, tuple(return_ids))

        return action
