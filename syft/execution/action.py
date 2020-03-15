from abc import ABC


class Action(ABC):
    """Describes the concrete steps workers can take with objects they own

    In Syft, an Action is when one worker wishes to tell another worker to do something with
    objects contained in the worker._objects registry (or whatever the official object store is
    backed with in the case that it's been overridden). For example, telling a worker to take two
    tensors and add them together is an Action. Sending an object from one worker to another is
    also an Action."""

    pass
