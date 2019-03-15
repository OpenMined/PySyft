import syft
from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
from syft.frameworks.torch.tensors.interpreters.utils import hook


class LoggingTensor(AbstractTensor):
    def __init__(
        self, parent: AbstractTensor = None, owner=None, id=None, tags=None, description=None
    ):
        """Initializes a LoggingTensor, whose behaviour is to log all operations
        applied on it.

        Args:
            parent: An optional AbstractTensor wrapper around the LoggingTensor
                which makes it so that you can pass this LoggingTensor to all
                the other methods/functions that PyTorch likes to use, although
                it can also be other tensors which extend AbstractTensor, such
                as custom tensors for Secure Multi-Party Computation or
                Federated Learning.
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: An optional string or integer id of the LoggingTensor.
        """
        super().__init__(tags, description)

        self.parent = parent
        self.owner = owner
        self.id = id
        self.child = None

    @hook
    def add(self, _self, *args, **kwargs):
        print("Log add")
        response = getattr(_self, "add")(*args, **kwargs)

        return response

    def manual_add(self, *args, **kwargs):
        # Replace all syft tensor with their child attribute
        new_self, new_args, new_kwargs = syft.frameworks.torch.hook_args.hook_method_args(
            "add", self, args, kwargs
        )

        print("Log add")
        # Send it to the appropriate class and get the response
        response = getattr(new_self, "add")(*new_args, **new_kwargs)

        # Put back SyftTensor on the tensors found in the response
        response = syft.frameworks.torch.hook_args.hook_response(
            "add", response, wrap_type=type(self)
        )
        return response

    @classmethod
    def handle_func_command(cls, command):
        """
        Receive an instruction for a function to be applied on a LoggingTensor,
        Perform some specific action (like logging) which depends of the
        instruction content, replace in the args all the LogTensors with
        their child attribute, forward the command instruction to the
        handle_function_command of the type of the child attributes, get the
        response and replace a LoggingTensor on top of all tensors found in
        the response.
        :param command: instruction of a function command: (command name,
        <no self>, arguments[, kwargs])
        :return: the response of the function command
        """
        cmd, _, args, kwargs = command

        # Do what you have to
        print("Logtensor logging function", cmd)

        # TODO: I can't manage the import issue, can you?
        # Replace all LoggingTensor with their child attribute
        new_args, new_kwargs, new_type = syft.frameworks.torch.hook_args.hook_function_args(
            cmd, args, kwargs
        )

        # build the new command
        new_command = (cmd, None, new_args, new_kwargs)

        # Send it to the appropriate class and get the response
        response = new_type.handle_func_command(new_command)

        # Put back LoggingTensor on the tensors found in the response
        response = syft.frameworks.torch.hook_args.hook_response(cmd, response, wrap_type=cls)

        return response
