import torch
from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
from syft.frameworks.torch.tensors.interpreters.utils import hook


class AdditiveSharingTensor(AbstractTensor):
    def __init__(
        self,
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
        self.child = None

        self.field = (2 ** 31) - 1 if field is None else field  # < 63 bits
        self.crypto_provider = crypto_provider

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

    def abstract_mul(self, equation: str, shares: dict, other_shares, **kwargs):
        """Abstractly Multiplies two tensors

        Args:
            equation: a string reprsentation of the equation to be computed in einstein
                summation form
            shares: a dictionary <location_id -> PointerTensor) of shares corresponding to
                self. Equivalent to calling self.child.
            other_shares: a dictionary <location_id -> PointerTensor) of shares corresponding
                to the tensor being multiplied by self.
        """

        # if someone passes in a constant... (i.e., x + 3)
        if not isinstance(other_shares, dict):
            other_shares = torch.Tensor([other_shares]).share(*self.child.keys()).child

        assert len(shares) == len(other_shares)

        if self.crypto_provider is None:
            raise AttributeError("For multiplication a crytoprovider must be passed.")

        locations = []
        for location, pointer in shares.items():
            locations.append(location)

        triple = self.crypto_provider.generate_triple(
            equation,
            self.field,
            tuple(shares[locations[0]].shape),
            tuple(other_shares[locations[0]].shape),
            locations,
        )
        a, b, c = triple

        d = {}
        e = {}
        for location in locations:
            d[location] = (shares[location] - a[location]) % self.field
            e[location] = (other_shares[location] - b[location]) % self.field

        delta = torch.zeros(d.items()[0].shape)
        epsilson = torch.zeros(e.items()[0].shape)

        for (location,) in locations:
            d_temp = d[location].get()
            e_temp = e[location].get()
            delta = delta + d_temp
            epsilson = epsilson + e_temp

        delta_epsilon = torch.einsum(equation, delta, epsilon)

        delta_ptrs = {}
        epsilon_ptrs = {}
        a_epsilon = {}
        delta_b = {}
        z = {}
        for location in locations:
            delta_ptrs[location] = delta.send(location)
            epsilon_ptrs[location] = epsilon.send(location)
            a_epsilon[location] = torch.einsum(equation, a[location], epsilon_ptrs[location])
            delta_b[location] = torch.einsum(equation, delta_ptrs[location], b[location])
            z[location] = a_epsilon[location] + delta_b[location] + c[location]
        delta_epsilon_pointer = delta_epsilon.send(locations[0])
        z[locations[0]] = z[locations[0]] + delta_epsilon_pointer

        return AdditiveSharingTensor(
            field=self.field, crypto_provider=self.crypto_provider
        ).set_shares(z)

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

        return self.abstract_mul("ij,ij->ij", shares, other_shares, **kwargs)

    def __mul__(self, *args, **kwargs):
        """Multiplies two number for details see mul
        """
        return self.mul(self, *args, **kwargs)

    @hook
    def matmul(self, shares: dict, other_shares, *args, **kwargs):
        """Multiplies two tensors together
        For details see abstract_mul

        Args:
            shares: a dictionary <location_id -> PointerTensor) of shares corresponding to
                self. Equivalent to calling self.child.
            other_shares: a dictionary <location_id -> PointerTensor) of shares corresponding
                to the tensor being multiplied by self.
        """

        return self.abstract_mul("ij,jk->ik", shares, other_shares, **kwargs)

    def __matmul__(self, *args, **kwargs):
        """Multiplies two number for details see mul
        """
        return self.mul(self, *args, **kwargs)
