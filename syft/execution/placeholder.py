import syft
from syft.generic.frameworks.hook import hook_args
from syft.execution.placeholder_id import PlaceholderId
from syft.generic.tensor import AbstractTensor
from syft.workers.abstract import AbstractWorker
from syft_proto.execution.v1.placeholder_pb2 import Placeholder as PlaceholderPB


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
        super().__init__(id=id, owner=owner, tags=tags, description=description)

        if not isinstance(self.id, PlaceholderId):
            self.id = PlaceholderId(self.id)
        self.child = None

    def instantiate(self, tensor):
        """
        Add a tensor as a child attribute. All operations on the placeholder will be also
        executed on this child tensor.

        We remove wrappers or Placeholders if is there are any.
        """
        if isinstance(tensor, PlaceHolder):
            self.child = tensor.child
        elif tensor.is_wrapper:
            self.instantiate(tensor.child)
        else:
            self.child = tensor
        return self

    def __str__(self) -> str:
        """
        Compact representation of a Placeholder, including tags and optional child
        """
        tags = " ".join(list(self.tags or []))

        out = f"{type(self).__name__ }[Tags:{tags}]"

        if hasattr(self, "child") and self.child is not None:
            out += f">{self.child}"

        return out

    __repr__ = __str__

    def copy(self):
        """
        Copying a placeholder doesn't duplicate the child attribute, because all
        copy operations happen locally where we want to keep reference to the same
        instantiated object. As the child doesn't get sent, this is not an issue.
        """
        placeholder = PlaceHolder(tags=self.tags, owner=self.owner)
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

        return PlaceHolder(owner=worker, id=tensor_id, tags=tags, description=description)

    @staticmethod
    def bufferize(worker: AbstractWorker, tensor: "PlaceHolder") -> PlaceholderPB:
        """Takes the attributes of a PlaceHolder and saves them in a Protobuf message.

        Args:
            worker: the worker doing the serialization
            tensor: a PlaceHolder.

        Returns:
            PlaceholderPB: a Protobuf message holding the unique attributes of the PlaceHolder.
        """

        protobuf_placeholder = PlaceholderPB()
        syft.serde.protobuf.proto.set_protobuf_id(protobuf_placeholder.id, tensor.id.value)
        protobuf_placeholder.tags.extend(tensor.tags)

        if tensor.description:
            protobuf_placeholder.description = tensor.description

        return protobuf_placeholder

    @staticmethod
    def unbufferize(worker: AbstractWorker, protobuf_placeholder: PlaceholderPB) -> "PlaceHolder":
        """
            This function reconstructs a PlaceHolder given it's attributes in form of a Protobuf message.
            Args:
                worker: the worker doing the deserialization
                protobuf_placeholder: a Protobuf message holding the attributes of the PlaceHolder
            Returns:
                PlaceHolder: a PlaceHolder
            """

        tensor_id = syft.serde.protobuf.proto.get_protobuf_id(protobuf_placeholder.id)
        tags = set(protobuf_placeholder.tags)

        description = None
        if bool(protobuf_placeholder.description):
            description = protobuf_placeholder.description

        return PlaceHolder(owner=worker, id=tensor_id, tags=tags, description=description)


### Register the tensor with hook_args.py ###
hook_args.default_register_tensor(PlaceHolder)
