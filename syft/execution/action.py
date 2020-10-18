from abc import ABC
from abc import abstractmethod

import syft as sy
from syft.execution.placeholder import PlaceHolder
from syft.execution.placeholder_id import PlaceholderId
from syft.generic.abstract.syft_serializable import SyftSerializable
from syft.workers.abstract import AbstractWorker


class Action(ABC, SyftSerializable):
    """Describes the concrete steps workers can take with objects they own

    In Syft, an Action is when one worker wishes to tell another worker to do something with
    objects contained in the worker.object_store registry (or whatever the official object store is
    backed with in the case that it's been overridden). For example, telling a worker to take two
    tensors and add them together is an Action. Sending an object from one worker to another is
    also an Action."""

    def __init__(
        self, name: str, target, args_: tuple, kwargs_: dict, return_ids: tuple, return_value=False
    ):
        """Initialize an action

        Args:
            name (String): The name of the method to be invoked (e.g. "__add__")
            target (Tensor): The object to invoke the method on
            args_ (Tuple): The arguments to the method call
            kwargs_ (Dictionary): The keyword arguments to the method call
            return_ids (Tuple): primarily for our async infrastructure (Plan, Protocol, etc.),
                the id of action results are set by the client. This allows the client to be able to
                predict where the results will be ahead of time. Importantly, this allows the
                client to pre-initalize the pointers to the future data, regardless of whether
                the action has yet executed. It also reduces the size of the response from the
                action (which is very often empty).
            return_value (boolean): return the result or not. If true, the result is directly
                returned, if not, the command sender will create a pointer to the remote result
                using the return_ids and will need to do .get() later to get the result.

        """

        # call the parent constructor - setting the type integer correctly
        super().__init__()

        self.name = name
        self.target = target
        self.args = args_
        self.kwargs = kwargs_
        self.return_ids = return_ids
        self.return_value = return_value

        self._type_check("name", str)
        self._type_check("args", tuple)
        self._type_check("kwargs", dict)
        self._type_check("return_ids", tuple)

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.target == other.target
            and self.args == other.args
            and self.kwargs == other.kwargs
            and self.return_ids == other.return_ids
        )

    def code(self, var_names=None) -> str:
        """Returns pseudo-code representation of computation action"""

        def stringify(obj, unroll_lists=True):
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
            elif isinstance(obj, (tuple, list)) and unroll_lists:
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
        if len(self.args) > 0:
            out += ", ".join([stringify(arg, unroll_lists=False) for arg in self.args])
        if self.kwargs:
            if len(self.args) > 0:
                out += ", "
            out += ", ".join(f"{k}={w}" for k, w in self.kwargs.items())
        out += ")"

        return out

    def __str__(self) -> str:
        """Returns string representation of this action"""
        return f"{type(self).__name__}[{self.code()}]"

    def _type_check(self, field_name, expected_type):
        actual_value = getattr(self, field_name)
        assert actual_value is None or isinstance(actual_value, expected_type), (
            f"{field_name} must be {expected_type.__name__}, but was "
            f"{type(actual_value).__name__}: {actual_value}."
        )

    # These methods must be implemented by child classes in order to return the correct type
    # and to be detected by the serdes as serializable. They are therefore marked as abstract
    # methods even though implementations are provided. Child classes may delegate to these
    # implementations, but must implement their own conversions to the appropriate classes.

    @staticmethod
    @abstractmethod
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
        return (
            sy.serde.msgpack.serde._simplify(worker, action.name),
            sy.serde.msgpack.serde._simplify(worker, action.target),
            sy.serde.msgpack.serde._simplify(worker, action.args),
            sy.serde.msgpack.serde._simplify(worker, action.kwargs),
            sy.serde.msgpack.serde._simplify(worker, action.return_ids),
            sy.serde.msgpack.serde._simplify(worker, action.return_value),
        )

    @staticmethod
    @abstractmethod
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
        name, target, args_, kwargs_, return_ids, return_value = action_tuple

        return (
            sy.serde.msgpack.serde._detail(worker, name),
            sy.serde.msgpack.serde._detail(worker, target),
            sy.serde.msgpack.serde._detail(worker, args_),
            sy.serde.msgpack.serde._detail(worker, kwargs_),
            sy.serde.msgpack.serde._detail(worker, return_ids),
            sy.serde.msgpack.serde._detail(worker, return_value),
        )

    @staticmethod
    @abstractmethod
    def bufferize(worker: AbstractWorker, action: "Action", protobuf_action):
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
        protobuf_action.command = action.name

        protobuf_target = None
        if isinstance(action.target, sy.generic.pointers.pointer_tensor.PointerTensor):
            protobuf_target = protobuf_action.target_pointer
        elif isinstance(action.target, sy.execution.placeholder_id.PlaceholderId):
            protobuf_target = protobuf_action.target_placeholder_id
        elif isinstance(action.target, (int, str)):
            sy.serde.protobuf.proto.set_protobuf_id(protobuf_action.target_id, action.target)
        elif action.target is not None:
            protobuf_target = protobuf_action.target_tensor

        if protobuf_target is not None:
            protobuf_target.CopyFrom(sy.serde.protobuf.serde._bufferize(worker, action.target))

        if action.args:
            protobuf_action.args.extend(sy.serde.protobuf.serde.bufferize_args(worker, action.args))

        if action.kwargs:
            for key, value in action.kwargs.items():
                protobuf_action.kwargs.get_or_create(key).CopyFrom(
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
                    protobuf_action.return_placeholder_ids.append(
                        sy.serde.protobuf.serde._bufferize(worker, return_id)
                    )
                else:
                    sy.serde.protobuf.proto.set_protobuf_id(
                        protobuf_action.return_ids.add(), return_id
                    )

        return protobuf_action

    @staticmethod
    @abstractmethod
    def unbufferize(worker: AbstractWorker, protobuf_obj):
        """
        This function takes the Protobuf version of this message and converts
        it into an Action. The bufferize() method runs the inverse of this method.

        Args:
            worker (AbstractWorker): a reference to the worker necessary for detailing. Read
                syft/serde/serde.py for more information on why this is necessary.
            protobuf_obj (ActionPB): the Protobuf message

        Returns:
            obj (tuple): a tuple of the args required to instantiate an Action object

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

        return_ids = tuple(
            sy.serde.protobuf.proto.get_protobuf_id(pb_id) for pb_id in protobuf_obj.return_ids
        )

        return_placeholder_ids = tuple(
            (
                sy.serde.protobuf.serde._unbufferize(worker, placeholder)
                for placeholder in protobuf_obj.return_placeholder_ids
            )
        )

        if return_placeholder_ids:
            action = (command, target, args_, kwargs_, return_placeholder_ids)
        else:
            action = (command, target, args_, kwargs_, return_ids)

        return action
