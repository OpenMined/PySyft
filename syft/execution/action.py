from abc import ABC
from abc import abstractmethod

from syft.execution.placeholder import PlaceHolder
from syft.execution.placeholder_id import PlaceholderId


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
