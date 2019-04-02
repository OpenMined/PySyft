import torch
import syft as sy
from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
from syft.frameworks.torch.overload_torch import overloaded

# Crypto protocols
from syft.frameworks.torch.crypto import spdz
from syft.frameworks.torch.crypto import securenn


class AdditiveSharingTensor(AbstractTensor):
    def __init__(
        self,
        shares: dict = None,
        parent: AbstractTensor = None,
        owner=None,
        id=None,
        field=None,
        n_bits=None,
        crypto_provider=None,
        tags=None,
        description=None,
    ):
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
        super().__init__(id=id, owner=owner, tags=tags, description=description, parent=parent)

        self.child = shares

        self.field = (2 ** 31) if field is None else field  # < 63 bits
        self.n_bits = 31 if n_bits is None else n_bits  # < 63 bits
        assert 2 ** self.n_bits == self.field
        self.crypto_provider = crypto_provider

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        type_name = type(self).__name__
        out = f"[" f"{type_name}]"
        for v in self.child.values():
            out += "\n\t-> " + str(v)
        if self.crypto_provider is not None:
            out += "\n\t*crypto provider: {}*".format(self.crypto_provider.id)
        return out

    @property
    def location(self):
        """Provide a location attribute"""
        return [s.owner for s in self.child.values()]

    @property
    def shape(self):
        """
        Return the shape which is the shape of any of the shares
        """
        for share in self.child.values():
            return share.shape

    def get_class_attributes(self):
        """
        Specify all the attributes need to build a wrapper correctly when returning a response,
        for example precision_fractional is important when wrapping the result of a method
        on a self which is a fixed precision tensor with a non default precision_fractional.
        """
        return {"crypto_provider": self.crypto_provider, "field": self.field, "n_bits": self.n_bits}

    def get(self):
        """Fetches all shares and returns the plaintext tensor they represent"""

        shares = list()

        for v in self.child.values():
            shares.append(v.get())

        return sum(shares)

    def virtual_get(self):
        """Get the value of the tensor without calling get - Only for VirtualWorkers"""

        shares = list()

        for v in self.child.values():
            share = v.location._objects[v.id_at_location]
            shares.append(share)

        return sum(shares)

    def init_shares(self, *owners):
        """Initializes shares and distributes them amongst their respective owners

        Args:
            *owners the list of shareholders. Can be of any length.

            """
        shares = self.generate_shares(
            self.child, n_workers=len(owners), field=self.field, random_type=torch.LongTensor
        )

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
        assert isinstance(
            shares, dict
        ), "Share should be provided as a dict {'worker.id': pointer, ...}"
        self.child = shares
        return self

    @staticmethod
    def generate_shares(secret, n_workers, field, random_type):
        """The cryptographic method for generating shares given a secret tensor.

        Args:
            secret: the tensor to be shared.
            n_workers: the number of shares to generate for each value
                (i.e., the number of tensors to return)
            field: 1 + the max value for a share
            random_type: the torch type shares should be encoded in (use the smallest possible
                given the choise of mod"
            """

        if not isinstance(secret, random_type):
            secret = secret.type(random_type)

        random_shares = [random_type(secret.shape) for i in range(n_workers - 1)]

        for share in random_shares:
            share.random_(field)

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

    @overloaded.overload_method
    def _getitem_multipointer(self, _self_shares, indices_shares):
        selected_shares = {}
        for worker, share in _self_shares.items():
            indices = []
            for index in indices_shares:
                if isinstance(index, slice):
                    indices.append(index)
                elif isinstance(index, dict):
                    indices.append(index[worker])
                else:
                    raise NotImplementedError("Index type", type(indices), "not supported")
            selected_share = share[tuple(indices)]
            selected_shares[worker] = selected_share

        return selected_shares

    def __getitem__(self, indices):
        tensor_type = type(indices)
        if isinstance(indices, tuple):
            for index in indices:
                if isinstance(index, AbstractTensor):
                    tensor_type = type(index)

        if tensor_type == sy.MultiPointerTensor:
            return self._getitem_multipointer(indices)
        else:
            raise NotImplementedError("Index type", type(indices), "not supported")

    ## SECTION SPDZ

    @overloaded.method
    def add(self, shares: dict, other_shares, *args, **kwargs):
        """Adds two tensors together

        Args:
            shares: a dictionary <location_id -> PointerTensor) of shares corresponding to
                self. Equivalent to calling self.child.
            other_shares: a dictionary <location_id -> PointerTensor) of shares corresponding
                to the tensor being added to self.
        """

        # if someone passes in a constant... (i.e., x + 3)
        if not isinstance(other_shares, dict):
            other_shares = torch.Tensor([other_shares]).share(*self.child.keys()).child.child

        assert len(shares) == len(other_shares)

        # matches each share which needs to be added according
        # to the location of the share
        new_shares = {}
        for k, v in shares.items():
            new_shares[k] = other_shares[k] + v

        return new_shares

    def __add__(self, other, **kwargs):
        """Adds two tensors. Forwards command to add. See add() for more details."""

        return self.add(other, **kwargs)

    @overloaded.method
    def sub(self, shares: dict, other_shares, **kwargs):
        """Subtracts an other tensor from self.

        Args:
            shares: a dictionary <location_id -> PointerTensor) of shares corresponding to
                self. Equivalent to calling self.child.
            other_shares: a dictionary <location_id -> PointerTensor) of shares corresponding
                to the tensor being subtracted from self.
        """

        # if someone passes in a constant... (i.e., x - 3), make it a shared tensor and keep the dict
        if not isinstance(other_shares, dict):
            other_shares = torch.Tensor([other_shares]).share(*self.child.keys()).child.child

        assert len(shares) == len(other_shares)

        # matches each share which needs to be added according
        # to the location of the share
        new_shares = {}
        for k, v in shares.items():
            new_shares[k] = v - other_shares[k]

        return new_shares

    def __sub__(self, *args, **kwargs):
        """Subtracts two tensors. Forwards command to sub. See .sub() for details."""
        return self.sub(*args, **kwargs)

    def _abstract_mul(self, equation: str, shares: dict, other_shares, **kwargs):
        """Abstractly Multiplies two tensors

        Args:
            equation: a string reprsentation of the equation to be computed in einstein
                summation form
            shares: a dictionary <location_id -> PointerTensor) of shares corresponding to
                self. Equivalent to calling self.child.
            other_shares: a dictionary <location_id -> PointerTensor) of shares corresponding
                to the tensor being multiplied by self.
        """
        # check to see that operation is either mul or matmul
        assert equation == "mul" or equation == "matmul"
        cmd = getattr(torch, equation)

        # if someone passes in a constant... (i.e., x + 3)
        # TODO: Handle public mul more efficiently
        if not isinstance(other_shares, dict):
            other_shares = torch.Tensor([other_shares]).share(*self.child.keys()).child.child

        assert len(shares) == len(other_shares)

        if self.crypto_provider is None:
            raise AttributeError("For multiplication a crytoprovider must be passed.")

        shares = spdz.spdz_mul(cmd, shares, other_shares, self.crypto_provider, self.field)

        return shares

    @overloaded.method
    def mul(self, shares: dict, other_shares, *args, **kwargs):
        """Multiplies two tensors together
        For details see abstract_mul

        Args:
            shares: a dictionary <location_id -> PointerTensor) of shares corresponding to
                self. Equivalent to calling self.child.
            other_shares: a dictionary <location_id -> PointerTensor) of shares corresponding
                to the tensor being multiplied by self.
        """

        return self._abstract_mul("mul", shares, other_shares, **kwargs)

    def __mul__(self, other, **kwargs):
        """Multiplies two number for details see mul
        """
        if isinstance(other, sy.MultiPointerTensor):
            return self.mul(other, **kwargs) / len(other.child.keys())
        return self.mul(other, **kwargs)

    @overloaded.method
    def matmul(self, shares: dict, other_shares, **kwargs):
        """Multiplies two tensors together
        For details see abstract_mul

        Args:
            shares: a dictionary <location_id -> PointerTensor) of shares corresponding to
                self. Equivalent to calling self.child.
            other_shares: a dictionary <location_id -> PointerTensor) of shares corresponding
                to the tensor being multiplied by self.
        """
        return self._abstract_mul("matmul", shares, other_shares, **kwargs)

    def mm(self, *args, **kwargs):
        """Multiplies two number for details see mul
        """
        return self.matmul(*args, **kwargs)

    def __matmul__(self, *args, **kwargs):
        """Multiplies two number for details see mul
        """
        return self.matmul(*args, **kwargs)

    def __itruediv__(self, *args, **kwargs):

        result = self.__truediv__(*args, **kwargs)
        self.child = result.child

    @overloaded.method
    def __truediv__(self, shares: dict, divisor):
        assert isinstance(divisor, int)

        divided_shares = {}
        for location, pointer in shares.items():
            divided_shares[location] = pointer / divisor

        return divided_shares

    @overloaded.method
    def mod(self, shares: dict, modulus: int):
        assert isinstance(modulus, int)

        moded_shares = {}
        for location, pointer in shares.items():
            moded_shares[location] = pointer % modulus

        return moded_shares

    def __mod__(self, *args, **kwargs):
        return self.mod(*args, **kwargs)

    ## SECTION SNN

    def relu(self):
        return securenn.relu(self)

    def positive(self):
        # self >= 0
        return securenn.relu_deriv(self)

    def gt(self, other):
        r = self - other - 1
        return r.positive()

    def __gt__(self, other):
        return self.gt(other)

    def ge(self, other):
        return (self - other).positive()

    def __ge__(self, other):
        return self.ge(other)

    def lt(self, other):
        return (other - self - 1).positive()

    def __lt__(self, other):
        return self.lt(other)

    def le(self, other):
        return (other - self).positive()

    def __le__(self, other):
        return self.le(other)

    def eq(self, other):
        diff = self - other
        diff2 = diff * diff
        negdiff2 = diff2 * -1
        return negdiff2.positive()

    def __eq__(self, other):
        return self.eq(other)

    ## STANDARD

    @staticmethod
    def dispatch(args, worker):
        """
        utility function for handle_func_command which help to select
        shares (seen as elements of dict) in an argument set. It could
        perhaps be put elsewhere

        Args:
            args: arguments to give to a functions
            worker: owner of the shares to select

        Return:
            args where the AdditiveSharedTensors are replaced by
            the appropriate share
        """
        return map(lambda x: x[worker] if isinstance(x, dict) else x, args)

    @classmethod
    def handle_func_command(cls, command):
        """
        Receive an instruction for a function to be applied on a Syft Tensor,
        Replace in the args all the LogTensors with
        their child attribute, forward the command instruction to the
        handle_function_command of the type of the child attributes, get the
        response and replace a Syft Tensor on top of all tensors found in
        the response.

        Args:
            command: instruction of a function command: (command name,
            <no self>, arguments[, kwargs])

        Returns:
            the response of the function command
        """
        cmd, _, args, kwargs = command

        tensor = args[0]

        # Check that the function has not been overwritten
        try:
            # Try to get recursively the attributes in cmd = "<attr1>.<attr2>.<attr3>..."
            cmd = cls.rgetattr(cls, cmd)
            return cmd(*args, **kwargs)
        except AttributeError:
            pass

        # TODO: I can't manage the import issue, can you?
        # Replace all LoggingTensor with their child attribute
        new_args, new_kwargs, new_type = sy.frameworks.torch.hook_args.hook_function_args(
            cmd, args, kwargs
        )

        results = {}
        for worker, share in new_args[0].items():
            new_type = type(share)
            new_args = tuple(AdditiveSharingTensor.dispatch(new_args, worker))

            # build the new command
            new_command = (cmd, None, new_args, new_kwargs)

            # Send it to the appropriate class and get the response
            results[worker] = new_type.handle_func_command(new_command)

        # Put back AdditiveSharingTensor on the tensors found in the response
        response = sy.frameworks.torch.hook_args.hook_response(
            cmd, results, wrap_type=cls, wrap_args=tensor.get_class_attributes()
        )

        return response
