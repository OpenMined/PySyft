from itertools import zip_longest

import syft
from syft.generic.frameworks.hook import hook_args
from syft.generic.abstract.tensor import AbstractTensor
from syft.workers.abstract import AbstractWorker
from syft_proto.execution.v1.placeholder_pb2 import Placeholder as PlaceholderPB


class PlaceHolder(AbstractTensor):
    def __init__(
        self,
        role=None,
        tracing=False,
        id=None,
        tags: set = None,
        description: str = None,
        shape=None,
        expected_dtype=None,
    ):
        """A PlaceHolder acts as a tensor but does nothing special. It can get
        "instantiated" when a real tensor is appended as a child attribute. It
        will send forward all the commands it receives to its child tensor.

        When you send a PlaceHolder, you don't sent the instantiated tensors.

        Args:
            id: An optional string or integer id of the PlaceHolder.
        """
        super().__init__(id=id, tags=tags, description=description)

        if not isinstance(self.id, syft.execution.placeholder_id.PlaceholderId):
            self.id = syft.execution.placeholder_id.PlaceholderId(self.id)

        self.expected_shape = tuple(shape) if shape is not None else None
        self.expected_dtype = expected_dtype
        self.child = None
        self.role = role
        self.tracing = tracing

    def get_class_attributes(self):
        """
        Specify all the attributes need to build a wrapper correctly when returning a response.
        """
        return {"role": self.role, "tracing": self.tracing}

    @classmethod
    def handle_func_command(cls, command):
        """Receive an instruction for a function to be applied on a Placeholder,
        Replace in the args with their child attribute, forward the command
        instruction to the handle_function_command of the type of the child attributes,
        get the response and wrap it in a Placeholder.
        We use this method to perform the tracing.

        Args:
            command: instruction of a function command: (command name,
                <no self>, arguments[, kwargs])

        Returns:
            the response of the function command
        """
        cmd, _, args, kwargs = command

        # Replace all PlaceHolders with their child attribute
        new_args, new_kwargs, new_type = hook_args.unwrap_args_from_function(cmd, args, kwargs)

        # build the new command
        new_command = (cmd, None, new_args, new_kwargs)

        # Send it to the appropriate class and get the response
        response = new_type.handle_func_command(new_command)

        # Find first placeholder in args
        template_placeholder = None
        for arg in args:
            if isinstance(arg, PlaceHolder):
                template_placeholder = arg

        placeholders = PlaceHolder.convert_to_placeholders(response, template_placeholder)

        if template_placeholder.tracing:
            template_placeholder.role.register_action(
                (command, placeholders), syft.execution.computation.ComputationAction
            )

        return placeholders

    @staticmethod
    def convert_to_placeholders(response, template_placeholder):
        """ Turn back response to PlaceHolders """
        if isinstance(response, (tuple, list)):

            placeholders = tuple(
                PlaceHolder.create_from(
                    r, role=template_placeholder.role, tracing=template_placeholder.tracing
                )
                for r in response
            )
        else:
            placeholders = PlaceHolder.create_from(
                response, role=template_placeholder.role, tracing=template_placeholder.tracing
            )

        return placeholders

    def instantiate(self, tensor):
        """
        Add a tensor as a child attribute. All operations on the placeholder will be also
        executed on this child tensor.

        We remove Placeholders if is there are any.
        """
        if isinstance(tensor, PlaceHolder):
            self.child = tensor.child
        else:
            self.child = tensor

        if hasattr(self.child, "shape"):
            self.expected_shape = tuple(self.child.shape)

        if hasattr(self.child, "dtype"):
            self.expected_dtype = self.child.dtype

        return self

    def __str__(self) -> str:
        """
        Compact representation of a Placeholder, including tags and optional child
        """
        tags = " ".join(list(self.tags or []))

        out = f"{type(self).__name__ }[Id:{self.id.value}]"

        if self.has_child() and self.child is not None:
            out += f">{self.child}"

        return out

    __repr__ = __str__

    def send(self, *args, **kwargs):
        """
        calls move on child & register_action to role
        """
        response = self.child.send(*args, **kwargs)
        placeholder = PlaceHolder.convert_to_placeholders(response, self)
        command = ("send", self, args, kwargs)
        self.role.register_action(
            (command, placeholder), syft.execution.communication.CommunicationAction
        )
        return placeholder

    def move(self, *args, **kwargs):
        """
        calls move on a pointer tensor & register_action to role
        """
        response = self.child.move(*args, **kwargs)
        placeholder = PlaceHolder.convert_to_placeholders(response, self)
        command = ("move", self, args, kwargs)
        self.role.register_action(
            (command, placeholder), syft.execution.communication.CommunicationAction
        )
        return placeholder

    def share(self, *args, **kwargs):
        """
        Send a command to remote worker to additively share a tensor via pointer tensor
        """
        response = self.child.share(*args, **kwargs)
        placeholder = PlaceHolder.convert_to_placeholders(response, self)
        command = ("share", self, args, kwargs)
        self.role.register_action(
            (command, placeholder), syft.execution.communication.CommunicationAction
        )
        return placeholder

    def fix_prec(self, *args, **kwargs):
        """
        sends command to remote worker to transform a tensor to fix_precision via pointer tensor
        """
        response = self.child.fix_prec(*args, **kwargs)
        placeholder = PlaceHolder.convert_to_placeholders(response, self)
        command = ("fix_prec", self, args, kwargs)
        self.role.register_action(
            (command, placeholder), syft.execution.computation.ComputationAction
        )
        return placeholder

    def mid_get(self, *args, **kwargs):
        response = self.child.mid_get(*args, **kwargs)
        placeholder = PlaceHolder.convert_to_placeholders(self.child, self)
        command = ("mid_get", self, args, kwargs)
        self.role.register_action(
            (command, placeholder), syft.execution.communication.CommunicationAction
        )
        return placeholder

    def remote_get(self, *args, **kwargs):
        """
        calls remote_get on child & register_action to role
        """
        response = self.child.remote_get(*args, **kwargs)
        placeholder = PlaceHolder.convert_to_placeholders(response, self)
        command = ("remote_get", self, args, kwargs)
        self.role.register_action(
            (command, placeholder), syft.execution.communication.CommunicationAction
        )
        return placeholder

    def remote_send(self, *args, **kwargs):
        """
        calls remote_send on child & register_action to role
        """
        response = self.child.remote_send(*args, **kwargs)
        placeholder = PlaceHolder.convert_to_placeholders(response, self)
        command = ("remote_send", self, args, kwargs)
        self.role.register_action(
            (command, placeholder), syft.execution.communication.CommunicationAction
        )
        return placeholder

    def share_(self, *args, **kwargs):
        """
        calls share_ on child & register_action to role
        """
        response = self.child.share_(*args, **kwargs)
        placeholder = PlaceHolder.convert_to_placeholders(response, self)
        command = ("share_", self, args, kwargs)
        self.role.register_action(
            (command, placeholder), syft.execution.communication.CommunicationAction
        )
        return placeholder

    def get(self, *args, **kwargs):
        """Requests the tensor/chain being pointed to, be serialized and return via child"""
        response = self.child.get(*args, **kwargs)
        placeholder = PlaceHolder.convert_to_placeholders(response, self)
        command = ("get", self, args, kwargs)
        self.role.register_action(
            (command, placeholder), syft.execution.communication.CommunicationAction
        )
        return placeholder

    def copy(self):
        """
        Copying a placeholder doesn't duplicate the child attribute, because all
        copy operations happen locally where we want to keep reference to the same
        instantiated object. As the child doesn't get sent, this is not an issue.
        """
        placeholder = PlaceHolder(
            role=self.role,
            tracing=self.tracing,
            tags=self.tags,
            shape=self.expected_shape,
            expected_dtype=self.expected_dtype,
        )
        placeholder.child = self.child

        if self.tracing:
            command = ("copy", self, (), {}), placeholder
            self.role.register_action(command, syft.execution.computation.ComputationAction)

        return placeholder

    @staticmethod
    def create_from(tensor, role=None, tracing=False):
        """Helper method to create a placeholder already
        instantiated with tensor.
        """
        return PlaceHolder(role=role, tracing=tracing).instantiate(tensor)

    @staticmethod
    def insert(tensor, after, role=None, tracing=False):
        """ Helper method to add a placeholder in the specific place of tensor chain. """
        current_level = tensor
        while not isinstance(current_level, after) and current_level is not None:
            current_level = getattr(current_level, "child", None)

        if current_level is None:
            raise RuntimeError(
                f"Cannot insert Placeholder, chain does not contain {after.__name__} tensor type."
            )

        child = getattr(current_level, "child", None)
        if child is None:
            raise RuntimeError(
                f"Cannot insert Placeholder, {after.__name__} does not wrap anything."
            )

        placeholder = PlaceHolder.create_from(child, role, tracing)
        current_level.child = placeholder
        return placeholder

    @staticmethod
    def extract(tensor):
        """ Helper method to find and return placeholder in the tensor chain. """
        current_level = tensor
        while not isinstance(current_level, PlaceHolder) and current_level is not None:
            current_level = getattr(current_level, "child", None)
        return current_level

    @staticmethod
    def recursive_extract(results):
        if isinstance(results, (tuple, list)):
            results_placeholders = tuple(
                PlaceHolder.recursive_extract(result) for result in results
            )
        else:
            results_placeholders = PlaceHolder.extract(results)

        return results_placeholders

    @staticmethod
    def create_placeholders(args_shape, args_dtypes=()):
        """Helper method to create a list of placeholders with shapes
        in args_shape.
        """
        # In order to support -1 value in shape to indicate any dimension
        # we map -1 to 1 for shape dimensions.
        # TODO: A more complex strategy could be used
        mapped_shapes = []
        for shape in args_shape:
            if list(filter(lambda x: x < -1, shape)):
                raise ValueError(f"Invalid shape {shape}")
            mapped_shapes.append(tuple(map(lambda y: 1 if y == -1 else y, shape)))

        return [
            syft.framework.hook.create_zeros(shape, dtype=dtype, requires_grad=False)
            for shape, dtype in zip_longest(mapped_shapes, args_dtypes)
        ]

    @staticmethod
    def instantiate_placeholders(obj, response):
        """
        Utility function to instantiate recursively an object containing placeholders
        with a similar object but containing tensors
        """
        if obj is not None:
            if isinstance(obj, PlaceHolder):
                obj.instantiate(response)
            elif isinstance(obj, (list, tuple)):
                for ph, rep in zip(obj, response):
                    PlaceHolder.instantiate_placeholders(ph, rep)
            else:
                raise ValueError(
                    f"Response of type {type(response)} is not supported in "
                    "Placeholder.instantiate."
                )

    @staticmethod
    def simplify(worker: AbstractWorker, placeholder: "PlaceHolder") -> tuple:
        """Takes the attributes of a PlaceHolder and saves them in a tuple.

        Args:
            worker: the worker doing the serialization
            placeholder: a PlaceHolder.

        Returns:
            tuple: a tuple holding the unique attributes of the PlaceHolder.
        """

        return (
            syft.serde.msgpack.serde._simplify(worker, placeholder.id),
            syft.serde.msgpack.serde._simplify(worker, placeholder.tags),
            syft.serde.msgpack.serde._simplify(worker, placeholder.description),
            syft.serde.msgpack.serde._simplify(worker, placeholder.expected_shape),
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

        tensor_id, tags, description, shape = tensor_tuple

        tensor_id = syft.serde.msgpack.serde._detail(worker, tensor_id)
        tags = syft.serde.msgpack.serde._detail(worker, tags)
        description = syft.serde.msgpack.serde._detail(worker, description)
        shape = syft.serde.msgpack.serde._detail(worker, shape)

        return PlaceHolder(id=tensor_id, tags=tags, description=description, shape=shape)

    @staticmethod
    def bufferize(worker: AbstractWorker, placeholder: "PlaceHolder") -> PlaceholderPB:
        """Takes the attributes of a PlaceHolder and saves them in a Protobuf message.

        Args:
            worker: the worker doing the serialization
            placeholder: a PlaceHolder.

        Returns:
            PlaceholderPB: a Protobuf message holding the unique attributes of the PlaceHolder.
        """

        protobuf_placeholder = PlaceholderPB()
        syft.serde.protobuf.proto.set_protobuf_id(protobuf_placeholder.id, placeholder.id.value)
        protobuf_placeholder.tags.extend(placeholder.tags)

        if placeholder.description:
            protobuf_placeholder.description = placeholder.description

        if placeholder.expected_shape:
            protobuf_placeholder.expected_shape.dims.extend(placeholder.expected_shape)

        return protobuf_placeholder

    @staticmethod
    def unbufferize(worker: AbstractWorker, protobuf_placeholder: PlaceholderPB) -> "PlaceHolder":
        """
        This function reconstructs a PlaceHolder given it's attributes in form of a
        Protobuf message.
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

        expected_shape = tuple(protobuf_placeholder.expected_shape.dims) or None

        return PlaceHolder(id=tensor_id, tags=tags, description=description, shape=expected_shape)

    @staticmethod
    def get_protobuf_schema() -> PlaceholderPB:
        return PlaceholderPB


### Register the tensor with hook_args.py ###
hook_args.default_register_tensor(PlaceHolder)
