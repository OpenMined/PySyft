import torch
from syft.frameworks.torch.tensors.abstract import AbstractTensor
from .utils import hook

class AdditiveSharingTensor(AbstractTensor):
    def __init__(self,
                 parent: AbstractTensor = None,
                 owner=None,
                 id=None,
                 Q_BITS = 31,
                 BASE = 2):
        """Initializes an Additive Sharing Tensor, whose behaviour is to split a
        single tensor into shares, distribute the shares amongst several machines,
        and then manage how those shares are used to compute various arithmetic
        functions.

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
            Q_BITS: the amount of memory for each number  (the exponent)
            BASE: the amount of memory for each number (the base)
        """
        self.parent = parent
        self.owner = owner
        self.id = id
        self.child = None

        self.BASE = BASE
        self.Q_BITS = Q_BITS
        self.field = (self.BASE ** Q_BITS) - 1  # < 63 bits

    def get(self):
        """Fetches all shares and returns the plaintext tensor they represent"""

        shares = list()

        for v in self.child.values():
            shares.append(v.get())

        return sum(shares)

    def init_shares(self, *owners):
        """Initializes shares and distributes them amongst their respective owners

        Args:
            *owners the list of shareholders. Can be of any length.

            """
        shares = self.generate_shares(self.child,
                                      n_workers=len(owners),
                                      mod=self.field,
                                      random_type=torch.LongTensor)

        for i in range(len(shares)):
            shares[i] = shares[i].send(owners[i])

        shares_dict = {}
        for i in range(len(shares)):
            shares_dict[shares[i].location.id] = shares[i]

        self.child = shares_dict
        return self

    def set_shares(self, shares):
        """A setter for the shares value. This is used primarily at the end of function calls.
        (see add()  below for an example).

        Args:
            shares: a dict of shares. Each key is the id of a worker on which the share exists. Each value
                is the PointerTensor corresponding to a share.
            """
        self.child = shares
        return self

    @staticmethod
    def generate_shares(secret, n_workers, mod, random_type):
        """The cryptographic method for generating shares given a secret tensor.

        Args:
            secret: the tensor to be shared.
            n_workers: the number of shares to generate for each value
                (i.e., the number of tensors to return)
            mod: 1 + the max value for a share
            random_type: the torch type shares should be encoded in (use the smallest possible
                given the choise of mod"
            """


        if (not isinstance(secret, random_type)):
            secret = secret.type(random_type)

        random_shares = [random_type(secret.shape) for i in range(n_workers - 1)]

        for share in random_shares:
            share.random_(mod)

        shares = []
        for i in range(n_workers):
            if i == 0:
                share = random_shares[i]
            elif i < n_workers - 1:
                share = random_shares[i] - random_shares[i - 1]
            else:
                share = secret - random_shares[i - 1]
            shares.append(share)

        return shares

    @hook
    def add(self, shares:dict, other_shares, *args, **kwargs):
        """Adds two tensors together

        Args:
            shares: a dictionary <location_id -> PointerTensor) of shares corresponding to
                self. Equivalent to calling self.child.
            other_shares: a dictionary <location_id -> PointerTensor) of shares corresponding
                to the tensor being added to self.
        """

        # if someone passes in a constant... (i.e., x + 3)
        if (not isinstance(other_shares, dict)):
            other_shares = torch.Tensor([other_shares]).share(*self.child.keys()).child

        assert len(shares) == len(other_shares)

        # matches each share which needs to be added according
        # to the location of the share
        new_shares =  {}
        for k,v in shares.items():
            new_shares[k] = other_shares[k] + v

        # return the true tensor (unwrapped - wrapping will happen
        # automatically if needed)
        response = AdditiveSharingTensor().set_shares(new_shares)

        return response

    @hook
    def __add__(self, *args, **kwargs):
        """Adds two tensors. See add() for more details."""
        return self.add(*args, **kwargs)

    @hook
    def sub(self, shares:dict, other_shares, **kwargs):
        """Subtracts an other tensor from self.

        Args:
            shares: a dictionary <location_id -> PointerTensor) of shares corresponding to
                self. Equivalent to calling self.child.
            other_shares: a dictionary <location_id -> PointerTensor) of shares corresponding
                to the tensor being subtracted from self.
        """

        # if someone passes in a constant... (i.e., x - 3)
        if (not isinstance(other_shares, dict)):
            other_shares = torch.Tensor([other_shares]).share(*self.child.keys()).child

        assert len(shares) == len(other_shares)

        # matches each share which needs to be added according
        # to the location of the share
        new_shares =  {}
        for k,v in shares.items():
            new_shares[k] = v - other_shares[k]

        # return the true tensor (unwrapped - wrapping will happen
        # automatically if needed)
        response = AdditiveSharingTensor().set_shares(new_shares)

        return response

    @hook
    def __sub__(self, *args, **kwargs):
        """Subtracts two tensors. See .sub() forr details."""
        return self.sub(*args, **kwargs)

    #
    # def manual_add(self, *args, **kwargs):
    #     # Replace all syft tensor with their child attribute
    #     new_self, new_args = syft.frameworks.torch.hook_args.hook_method_args("add", self, args)
    #
    #     print("Log add")
    #     # Send it to the appropriate class and get the response
    #     response = getattr(new_self, "add")(*new_args, **kwargs)
    #
    #     # Put back SyftTensor on the tensors found in the response
    #     response = syft.frameworks.torch.hook_args.hook_response(
    #         "add", response, wrap_type=type(self)
    #     )
    #     return response
    #
    # @classmethod
    # def handle_func_command(cls, command):
    #     """
    #     Receive an instruction for a function to be applied on a LoggingTensor,
    #     Perform some specific action (like logging) which depends of the
    #     instruction content, replace in the args all the LogTensors with
    #     their child attribute, forward the command instruction to the
    #     handle_function_command of the type of the child attributes, get the
    #     response and replace a LoggingTensor on top of all tensors found in
    #     the response.
    #     :param command: instruction of a function command: (command name,
    #     <no self>, arguments[, kwargs])
    #     :return: the response of the function command
    #     """
    #     # TODO: add kwargs in command
    #     cmd, _, args = command
    #
    #     # Do what you have to
    #     print("Logtensor logging function", cmd)
    #
    #     # TODO: I can't manage the import issue, can you?
    #     # Replace all LoggingTensor with their child attribute
    #     new_args, new_type = syft.frameworks.torch.hook_args.hook_function_args(cmd, args)
    #
    #     # build the new command
    #     new_command = (cmd, None, new_args)
    #
    #     # Send it to the appropriate class and get the response
    #     response = new_type.handle_func_command(new_command)
    #
    #     # Put back LoggingTensor on the tensors found in the response
    #     response = syft.frameworks.torch.hook_args.hook_response(cmd, response, wrap_type=cls)
    #
    #     return response
