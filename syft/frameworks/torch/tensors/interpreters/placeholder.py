import torch

import syft
from syft.generic.frameworks.hook import hook_args
from syft.generic.tensor import AbstractTensor
from syft.workers.abstract import AbstractWorker


class PlaceHolder(AbstractTensor):
    def __init__(self, owner=None, id=None, tags: set = None, description: str = None):
        """A PlaceHolder acts as a tensor but does nothing special. It can get
        "instantiated" when a real tensor is appended as a child attribute. It
        will send forward all the commands it receives to its child tensor.

        When you send a PlaceHolder, you don't sent the instantiated tensors.

        Args:
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: An optional string or integer id of the PlaceHolder.
        """
        super().__init__(tags=tags, description=description)

        self.owner = owner
        self.id = id if id else syft.ID_PROVIDER.pop()
        self.child = None

    def instantiate(self, tensor):
        """
        Add a tensor as a child attribute. All operations on the placeholder will be also
        executed on this child tensor

        We remove wrappers is there are any.
        """
        if hasattr(tensor, "child") and tensor.is_wrapper and tensor.child is not None:
            self.child = tensor.child
        else:
            self.child = tensor
        return self

    def get(self):
        """
        Returns any content that has been filled into the placeholder.

        This is roughly the opposite of instantiate().
        """
        return self.child

    def __str__(self) -> str:
        if isinstance(self.tags, set):
            tags = ", ".join(list(self.tags))
        elif self.tags is None:
            tags = "-"
        else:
            tags = self.tags
        if hasattr(self, "child") and self.child is not None:
            return f"{type(self).__name__ }({tags})>" + self.child.__str__()
        else:
            return f"{type(self).__name__ }({tags})"

    __repr__ = __str__

    def copy(self):
        """
        Copying a placeholder doesn't duplicate the child attribute, because all
        copy operations happen locally where we want to keep reference to the same
        instantiated object. As the child doesn't get sent, this is not an issue.
        """
        placeholder = PlaceHolder(tags=self.tags)
        placeholder.child = self.child
        return placeholder

    @staticmethod
    def simplify(worker: AbstractWorker, tensor: "PlaceHolder") -> tuple:
        """Takes the attributes of a PlaceHolder and saves them in a tuple.

        Args:
            worker: the worker doing the serialization
            tensor: a PlaceHolder.

        Returns:
            tuple: a tuple holding the unique attributes of the PlaceHolder.
        """

        return (
            syft.serde.msgpack.serde._simplify(worker, tensor.id),
            syft.serde.msgpack.serde._simplify(worker, tensor.tags),
            syft.serde.msgpack.serde._simplify(worker, tensor.description),
        )

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "PlaceHolder":
        """
            This function reconstructs a PlaceHolder given it's attributes in form of a tuple.
            Args:
                worker: the worker doing the deserialization
                tensor_tuple: a tuple holding the attributes of the PlaceHolder
            Returns:
                PlaceHolder: a PlaceHolder
            """

        tensor_id, tags, description = tensor_tuple

        tensor_id = syft.serde.msgpack.serde._detail(worker, tensor_id)
        tags = syft.serde.msgpack.serde._detail(worker, tags)
        description = syft.serde.msgpack.serde._detail(worker, description)

        unique_id = "-".join(tags)

        # NOTE having a global var here is not elegant, we can iterate on this
        if unique_id not in syft.hook.placeholders:
            tensor = PlaceHolder(owner=worker, id=tensor_id, tags=tags, description=description)
            syft.hook.placeholders[unique_id] = tensor

        return syft.hook.placeholders[unique_id]


### Register the tensor with hook_args.py ###
hook_args.default_register_tensor(PlaceHolder)
