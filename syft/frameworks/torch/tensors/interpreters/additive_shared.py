import torch
from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
from syft.frameworks.torch.tensors.interpreters.utils import hook
from syft.frameworks.torch.crypto.spdz import spdz_mul


class AdditiveSharingTensor(AbstractTensor):
    def __init__(
        self,
        shares: dict = None,
        parent: AbstractTensor = None,
        owner=None,
        id=None,
        field=None,
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
        super().__init__(tags, description)

        self.parent = parent
        self.owner = owner
        self.id = id
        self.child = shares

        self.field = (2 ** 31) - 1 if field is None else field  # < 63 bits
        self.crypto_provider = crypto_provider

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        type_name = type(self).__name__
        out = f"[" f"{type_name}]"
        for v in self.child.values():
            out += "\n\t-> " + str(v)
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

    @hook
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
            other_shares = torch.Tensor([other_shares]).share(*self.child.keys()).child

        assert len(shares) == len(other_shares)

        # matches each share which needs to be added according
        # to the location of the share
        new_shares = {}
        for k, v in shares.items():
            new_shares[k] = other_shares[k] + v

        # return the true tensor (unwrapped - wrapping will happen
        # automatically if needed)
        response = AdditiveSharingTensor().set_shares(new_shares)

        return response

    def __add__(self, *args, **kwargs):
        """Adds two tensors. Forwards command to add. See add() for more details."""
        return self.add(*args, **kwargs)

    @hook
    def sub(self, shares: dict, other_shares, **kwargs):
        """Subtracts an other tensor from self.

        Args:
            shares: a dictionary <location_id -> PointerTensor) of shares corresponding to
                self. Equivalent to calling self.child.
            other_shares: a dictionary <location_id -> PointerTensor) of shares corresponding
                to the tensor being subtracted from self.
        """

        # if someone passes in a constant... (i.e., x - 3)
        if not isinstance(other_shares, dict):
            other_shares = torch.Tensor([other_shares]).share(*self.child.keys()).child

        assert len(shares) == len(other_shares)

        # matches each share which needs to be added according
        # to the location of the share
        new_shares = {}
        for k, v in shares.items():
            new_shares[k] = v - other_shares[k]

        # return the true tensor (unwrapped - wrapping will happen
        # automatically if needed)
        response = AdditiveSharingTensor().set_shares(new_shares)

        return response

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
            other_shares = torch.Tensor([other_shares]).share(*self.child.keys()).child

        assert len(shares) == len(other_shares)

        if self.crypto_provider is None:
            raise AttributeError("For multiplication a crytoprovider must be passed.")

        shares = spdz_mul(cmd, shares, other_shares, self.crypto_provider, self.field)

        return AdditiveSharingTensor(
            field=self.field, crypto_provider=self.crypto_provider
        ).set_shares(shares)

    @hook
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

    def __mul__(self, *args, **kwargs):
        """Multiplies two number for details see mul
        """
        return self.mul(*args, **kwargs)

    @hook
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

    def __matmul__(self, *args, **kwargs):
        """Multiplies two number for details see mul
        """
        return self.matmul(*args, **kwargs)

    def __itruediv__(self, *args, **kwargs):

        result = self.__truediv__(*args, **kwargs)
        self.child = result.child

    @hook
    def __truediv__(self, shares: dict, divisor):
        assert isinstance(divisor, int)

        divided_shares = {}
        for location, pointer in shares.items():
            divided_shares[location] = pointer / divisor

        return AdditiveSharingTensor(
            field=self.field, crypto_provider=self.crypto_provider
        ).set_shares(divided_shares)

    @hook
    def mod(self, shares: dict, modulus: int):
        assert isinstance(modulus, int)
        moded_shares = {}
        for location, pointer in shares.items():
            moded_shares[location] = pointer % modulus

        return AdditiveSharingTensor(
            field=self.field, crypto_provider=self.crypto_provider
        ).set_shares(moded_shares)

    def __mod__(self, *args, **kwargs):
        return self.mod(*args, **kwargs)
