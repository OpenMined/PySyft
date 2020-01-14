import torch

import syft
from syft.generic.frameworks.hook import hook_args
from syft.generic.tensor import AbstractTensor
from syft.workers.abstract import AbstractWorker


class PlaceHolder(AbstractTensor):
    def __init__(
        self,
        owner=None,
        id=None,
        tags: set = None,
        description: str = None,
    ):
        """Initializes a PlaceHoled
        Args:
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: An optional string or integer id of the FixedPrecisionTensor.
        """
        super().__init__(tags=tags, description=description)

        self.owner = owner
        self.id = id if id else syft.ID_PROVIDER.pop()
        self.child = None

    def instanciate(self, tensor):
        self.child = tensor
        return self

    def __str__(self) -> str:
        if isinstance(self.tags, set):
            tags = ', '.join(list(self.tags))
        elif self.tags is None:
            tags = '-'
        else:
            tags = self.tags
        if hasattr(self, "child") and self.child is not None:
            return f"{type(self).__name__ }({tags})>" + self.child.__str__()
        else:
            return f"{type(self).__name__ }({tags})"

    __repr__ = __str__

    def get_class_attributes(self):
        """
        Specify all the attributes need to build a wrapper correctly when returning a response,
        for example precision_fractional is important when wrapping the result of a method
        on a self which is a fixed precision tensor with a non default precision_fractional.
        """
        return {}

    @classmethod
    def handle_func_command(cls, command):
        """
        Receive an instruction for a function to be applied on a FixedPrecision Tensor,
        Perform some specific action (like logging) which depends of the
        instruction content, replace in the args all the FPTensors with
        their child attribute, forward the command instruction to the
        handle_function_command of the type of the child attributes, get the
        response and replace a FixedPrecision on top of all tensors found in
        the response.
        :param command: instruction of a function command: (command name,
        <no self>, arguments[, kwargs])
        :return: the response of the function command
        """
        cmd, _, args, kwargs = command

        tensor = args[0] if not isinstance(args[0], (tuple, list)) else args[0][0]

        # Check that the function has not been overwritten
        try:
            # Try to get recursively the attributes in cmd = "<attr1>.<attr2>.<attr3>..."
            cmd = cls.rgetattr(cls, cmd)
            return cmd(*args, **kwargs)
        except AttributeError:
            pass

        # Replace all FixedPrecisionTensor with their child attribute
        new_args, new_kwargs, new_type = hook_args.unwrap_args_from_function(cmd, args, kwargs)

        # build the new command
        new_command = (cmd, None, new_args, new_kwargs)

        # Send it to the appropriate class and get the response
        response = new_type.handle_func_command(new_command)

        # Put back FixedPrecisionTensor on the tensors found in the response
        response = hook_args.hook_response(
            cmd, response, wrap_type=cls, wrap_args=tensor.get_class_attributes()
        )

        return response


    @staticmethod
    def simplify(worker: AbstractWorker, tensor: "FixedPrecisionTensor") -> tuple:
        """Takes the attributes of a FixedPrecisionTensor and saves them in a tuple.

        Args:
            worker: the worker doing the serialization
            tensor: a FixedPrecisionTensor.

        Returns:
            tuple: a tuple holding the unique attributes of the fixed precision tensor.
        """
        chain = None
        if hasattr(tensor, "child"):
            chain = syft.serde.msgpack.serde._simplify(worker, tensor.child)

        return (
            syft.serde.msgpack.serde._simplify(worker, tensor.id),
            syft.serde.msgpack.serde._simplify(worker, tensor.tags),
            syft.serde.msgpack.serde._simplify(worker, tensor.description),
            chain,
        )

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "FixedPrecisionTensor":
        """
            This function reconstructs a FixedPrecisionTensor given it's attributes in form of a tuple.
            Args:
                worker: the worker doing the deserialization
                tensor_tuple: a tuple holding the attributes of the FixedPrecisionTensor
            Returns:
                FixedPrecisionTensor: a FixedPrecisionTensor
            Examples:
                shared_tensor = detail(data)
            """

        tensor_id, tags, description, chain = tensor_tuple

        tensor = PlaceHolder(
            owner=worker,
            id=syft.serde.msgpack.serde._detail(worker, tensor_id),
            tags=syft.serde.msgpack.serde._detail(worker, tags),
            description=syft.serde.msgpack.serde._detail(worker, description),
        )

        if chain is not None:
            chain = syft.serde.msgpack.serde._detail(worker, chain)
            tensor.child = chain

        return tensor


### Register the tensor with hook_args.py ###
hook_args.default_register_tensor(PlaceHolder)
