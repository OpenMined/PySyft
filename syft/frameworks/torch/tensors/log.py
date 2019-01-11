import syft
from syft.frameworks.torch.tensors.abstract import AbstractTensor


class LogTensor(AbstractTensor):
    def __init__(self, parent: AbstractTensor = None, owner=None, id=None):
        """Initializes a LogTensor, whose behaviour is to log all operations
        applied on it.

        Args:
            parent: An optional AbstractTensor wrapper around the LogTensor
                which makes it so that you can pass this LogTensor to all
                the other methods/functions that PyTorch likes to use, although
                it can also be other tensors which extend AbstractTensor, such
                as custom tensors for Secure Multi-Party Computation or
                Federated Learning.
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: An optional string or integer id of the LogTensor.
        """
        self.parent = parent
        self.owner = owner
        self.id = id
        self.child = None

    @classmethod
    def handle_method_command(cls, command):
        """
        Receive an instruction for a method to be applied on a LogTensor,
        Perform some specific action (like logging) which depends of the
        instruction content, replace in the args all the LogTensors with
        their child attribute, forward the command instruction to the
        handle_method_command of the type of the child attributes, get the
        response and replace a LogTensor on top of all tensors found in
        the response.
        :param command: instruction of a method command: (command name,
        self of the method, arguments[, kwargs])
        :return: the response of the method command
        """
        # TODO: add kwargs in command
        cmd, self, args = command

        # Do what you have to
        print("Logtensor logging method", cmd)

        # TODO: I can't manage the import issue, can you?
        # Replace all LogTensor with their child attribute
        new_self, new_args = syft.frameworks.torch.hook_args.hook_method_args(cmd, self, args)

        # build the new command
        new_command = (cmd, new_self, new_args)

        # Send it to the appropriate class and get the response
        response = type(new_self).handle_method_command(new_command)

        # Put back LogTensor on the tensors found in the response
        response = syft.frameworks.torch.hook_args.hook_method_response(
            cmd, response, wrap_type=cls
        )

        return response
