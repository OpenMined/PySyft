from abc import ABC
from abc import abstractmethod

import syft as sy
from syft.execution.placeholder import PlaceHolder
from syft.execution.placeholder_id import PlaceholderId
from syft.workers.abstract import AbstractWorker


class Action(ABC):
    """Describes the concrete steps workers can take with objects they own

    In Syft, an Action is when one worker wishes to tell another worker to do something with
    objects contained in the worker._objects registry (or whatever the official object store is
    backed with in the case that it's been overridden). For example, telling a worker to take two
    tensors and add them together is an Action. Sending an object from one worker to another is
    also an Action."""

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
        """Returns string representation of this action"""
        return f"{type(self).__name__}[{self.code()}]"

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
        message = (action.name, action.target, action.args, action.kwargs)

        return (
            sy.serde.msgpack.serde._simplify(worker, message),
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
        message = action_tuple[0]
        return_ids = action_tuple[1]
        return_value = action_tuple[2]

        detailed_msg = sy.serde.msgpack.serde._detail(worker, message)
        detailed_ids = sy.serde.msgpack.serde._detail(worker, return_ids)
        detailed_return_value = sy.serde.msgpack.serde._detail(worker, return_value)

        name, target, args_, kwargs_ = detailed_msg

        return (name, target, args_, kwargs_, detailed_ids, detailed_return_value)
