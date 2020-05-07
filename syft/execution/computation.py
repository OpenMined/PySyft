import syft as sy
from syft.workers.abstract import AbstractWorker

from syft.execution.action import Action
from syft.execution.placeholder import PlaceHolder
from syft.execution.placeholder_id import PlaceholderId

from syft_proto.execution.v1.computation_action_pb2 import ComputationAction as ComputationActionPB


class ComputationAction(Action):
    """Describes mathematical operations performed on tensors"""

    def __init__(self, name, target, args_, kwargs_, return_ids, return_value=False):
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
            return_value (boolean): return the result or not. If true, the result is directly returned,
                if not, the command sender will create a pointer to the remote result using the return_ids
                and will need to do .get() later to get the result.

        """

        # call the parent constructor - setting the type integer correctly
        super().__init__()

        self.name = name
        self.target = target
        self.args = args_
        self.kwargs = kwargs_
        self.return_ids = return_ids
        self.return_value = return_value

    @property
    def contents(self):
        """Return a tuple with the contents of the action (backwards compatability)

        Some of our codebase still assumes that all message types have a .contents attribute. However,
        the contents attribute is very opaque in that it doesn't put any constraints on what the contents
        might be. Since we know this message is a action, we instead choose to store contents in two pieces,
        self.message and self.return_ids, which allows for more efficient simplification (we don't have to
        simplify return_ids because they are always a list of integers, meaning they're already simplified)."""

        message = (self.name, self.target, self.args, self.kwargs)

        return (message, self.return_ids, self.return_value)

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
            sy.serde.msgpack.serde._simplify(worker, action.return_value),
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
        return_value = msg_tuple[2]

        detailed_msg = sy.serde.msgpack.serde._detail(worker, message)
        detailed_ids = sy.serde.msgpack.serde._detail(worker, return_ids)
        detailed_return_value = sy.serde.msgpack.serde._detail(worker, return_value)

        name, target, args_, kwargs_ = detailed_msg

        return ComputationAction(name, target, args_, kwargs_, detailed_ids, detailed_return_value)

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

        protobuf_target = None
        if isinstance(action.target, sy.generic.pointers.pointer_tensor.PointerTensor):
            protobuf_target = protobuf_op.target_pointer
        elif isinstance(action.target, sy.execution.placeholder_id.PlaceholderId):
            protobuf_target = protobuf_op.target_placeholder_id
        elif isinstance(action.target, (int, str)):
            sy.serde.protobuf.proto.set_protobuf_id(protobuf_op.target_id, action.target)
        elif action.target is not None:
            protobuf_target = protobuf_op.target_tensor

        if protobuf_target is not None:
            protobuf_target.CopyFrom(sy.serde.protobuf.serde._bufferize(worker, action.target))

        if action.args:
            protobuf_op.args.extend(sy.serde.protobuf.serde.bufferize_args(worker, action.args))

        if action.kwargs:
            for key, value in action.kwargs.items():
                protobuf_op.kwargs.get_or_create(key).CopyFrom(
                    sy.serde.protobuf.serde.bufferize_arg(worker, value)
                )

        if action.return_ids is not None:
            if not isinstance(action.return_ids, (list, tuple)):
                return_ids = (action.return_ids,)
            else:
                return_ids = action.return_ids

            for return_id in return_ids:
                if isinstance(return_id, PlaceholderId):
                    # NOTE to know when we have a PlaceholderId, we store it
                    # in return_placeholder_ids and not in return_ids
                    protobuf_op.return_placeholder_ids.append(
                        sy.serde.protobuf.serde._bufferize(worker, return_id)
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
        args_ = sy.serde.protobuf.serde.unbufferize_args(worker, protobuf_obj.args)

        kwargs_ = {}
        for key in protobuf_obj.kwargs:
            kwargs_[key] = sy.serde.protobuf.serde.unbufferize_arg(worker, protobuf_obj.kwargs[key])

        return_ids = [
            sy.serde.protobuf.proto.get_protobuf_id(pb_id) for pb_id in protobuf_obj.return_ids
        ]

        return_placeholder_ids = [
            sy.serde.protobuf.serde._unbufferize(worker, placeholder)
            for placeholder in protobuf_obj.return_placeholder_ids
        ]

        if return_placeholder_ids:
            action = ComputationAction(
                command, target, tuple(args_), kwargs_, return_placeholder_ids
            )
        else:
            action = ComputationAction(command, target, tuple(args_), kwargs_, tuple(return_ids))

        return action

    def code(self, var_names=None) -> str:
        """Returns pseudo-code representation of computation action"""

        def stringify(obj):
            if isinstance(obj, PlaceholderId):
                id = obj.value
                if var_names is None:
                    ret = f"var_{id}"
                else:
                    if id in var_names:
                        ret = var_names[id]
                    else:
                        idx = sum("var_" in k for k in var_names.values())
                        name = f"var_{idx}"
                        var_names[id] = name
                        ret = name
            elif isinstance(obj, PlaceHolder):
                ret = stringify(obj.id)
            elif isinstance(obj, (tuple, list)):
                ret = ", ".join(stringify(o) for o in obj)
            else:
                ret = str(obj)

            return ret

        out = ""
        if self.return_ids is not None:
            out += stringify(self.return_ids) + " = "
        if self.target is not None:
            out += stringify(self.target) + "."
        out += self.name + "("
        out += stringify(self.args)
        if self.kwargs:
            if len(self.args) > 0:
                out += ", "
            out += ", ".join(f"{k}={w}" for k, w in self.kwargs.items())
        out += ")"

        return out

    def __str__(self) -> str:
        """Returns string representation of computation action"""
        return f"{type(self).__name__}[{self.code()}]"
