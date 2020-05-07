import math
import torch
import warnings

import syft as sy
from syft.frameworks.torch.mpc import crypto_protocol
from syft.frameworks.torch.mpc import spdz
from syft.frameworks.torch.mpc import securenn
from syft.frameworks.torch.mpc import fss
from syft.generic.utils import memorize

from syft.generic.tensor import AbstractTensor
from syft.generic.frameworks.hook import hook_args
from syft.generic.frameworks.overload import overloaded
from syft.workers.abstract import AbstractWorker

from syft_proto.frameworks.torch.tensors.interpreters.v1.additive_shared_pb2 import (
    AdditiveSharingTensor as AdditiveSharingTensorPB,
)
from syft_proto.types.syft.v1.id_pb2 import Id as IdPB

no_wrap = {"no_wrap": True}


class AdditiveSharingTensor(AbstractTensor):
    def __init__(
        self,
        shares: dict = None,
        owner=None,
        id=None,
        field=None,
        protocol="snn",
        dtype=None,
        crypto_provider=None,
        tags=None,
        description=None,
    ):
        """Initializes an Additive Sharing Tensor, whose behaviour is to split a
        single tensor into shares, distribute the shares amongst several machines,
        and then manage how those shares are used to compute various arithmetic
        functions.

        Args:

            shares: Optional dictionary with the shares already split
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: An optional string or integer id of the AdditiveSharingTensor.
            field: size of the arithmetic field in which the shares live
            dtype: dtype of the field in which shares live
            crypto_provider: an optional BaseWorker providing crypto elements
                such as Beaver triples
            tags: an optional set of hashtags corresponding to this tensor
                which this tensor should be searchable for
            description: an optional string describing the purpose of the
                tensor
        """
        super().__init__(id=id, owner=owner, tags=tags, description=description)

        self.child = shares
        self.dtype = dtype
        if dtype == "custom":
            if field is None:
                raise ValueError("Field cannot be None for custom dtype")
            self.field = field
            self.torch_dtype = torch.int32 if field <= 2 ** 32 else torch.int64
        elif dtype == "long" or dtype == "int64":
            self.field = 2 ** 64
            self.torch_dtype = torch.int64
            self.dtype = "long"
        elif dtype == "int" or dtype == "int32":
            self.field = 2 ** 32
            self.torch_dtype = torch.int32
            self.dtype = "int"

        else:
            if dtype is not None:
                raise ValueError("Invalid dtype value: " + dtype)
            warnings.warn("Use dtype instead of field")
            # Since n mod 0 is not defined
            if isinstance(field, int) and field > 0:
                if field <= 2 ** 32:
                    self.dtype = "int"
                    self.field = 2 ** 32
                    self.torch_dtype = torch.int32
                else:
                    self.dtype = "long"
                    self.field = 2 ** 64
                    self.torch_dtype = torch.int64
            else:
                warnings.warn("Default args selected")
                # Default args
                self.dtype = "long"
                self.field = 2 ** 64
                self.torch_dtype = torch.int64

        if shares is not None:
            self.child = {}
            for location, share in shares.items():
                if isinstance(share, sy.PointerTensor):
                    self.child[location] = share
                elif share.is_wrapper and isinstance(share.child, sy.PointerTensor):
                    self.child[location] = share.child
                else:
                    raise ValueError(
                        f"Shares should be a dict of Pointers, optionally wrapped, but got:\n{shares}"
                    )
        else:
            self.child = None

        self.n_bits = self.calculateBits(self.field)
        # assert 2 ** self.n_bits == self.field

        # min value for shares in field
        self._min_value = None
        # max value for shares in field
        self._max_value = None

        self.crypto_provider = (
            crypto_provider if crypto_provider is not None else sy.hook.local_worker
        )

        self.protocol = protocol

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        type_name = type(self).__name__
        out = f"[" f"{type_name}]"
        if self.child is not None:
            for v in self.child.values():
                out += "\n\t-> " + str(v)
        if self.crypto_provider is not None:
            out += f"\n\t*crypto provider: {self.crypto_provider.id}*"
        return out

    def __bool__(self):
        """Prevent evaluation of encrypted tensor"""
        raise ValueError(
            "Additive shared tensors can't be converted boolean values. You should decrypt it first."
        )

    @property
    def locations(self):
        """Provide a locations attribute"""
        return [s.location for s in self.child.values()]

    @property
    def shape(self):
        """
        Return the shape which is the shape of any of the shares
        """
        for share in self.child.values():
            return share.shape

    @property
    def min_value(self):
        if self._min_value is None:
            self._min_value = -(self.field // 2)
        return self._min_value

    @property
    def max_value(self):
        if self._max_value is None:
            self._max_value = (self.field - 1) // 2
        return self._max_value

    def dim(self):
        for share in self.child.values():
            return len(share.shape)

    def clone(self):
        """
        Clone should keep ids unchanged, contrary to copy
        """
        cloned_tensor = type(self)(**self.get_class_attributes())
        cloned_tensor.id = self.id
        cloned_tensor.owner = self.owner

        cloned_tensor.child = {location: share.clone() for location, share in self.child.items()}

        return cloned_tensor

    def get_class_attributes(self):
        """
        Specify all the attributes need to build a wrapper correctly when returning a response,
        for example precision_fractional is important when wrapping the result of a method
        on a self which is a fixed precision tensor with a non default precision_fractional.
        """
        return {
            "crypto_provider": self.crypto_provider,
            "dtype": self.dtype,
            "field": self.field,
            "protocol": self.protocol,
        }

    @property
    def grad(self):
        """
        Gradient makes no sense for Additive Shared Tensor, so we make it clear
        that if someone query .grad on a Additive Shared Tensor it doesn't error
        but returns grad and can't be set
        """
        return None

    def backward(self, *args, **kwargs):
        """Calling backward on Additive Shared Tensor doesn't make sense, but sometimes a call
        can be propagated downward the chain to an AST (for example in create_grad_objects), so
        we just ignore the call."""
        pass

    @staticmethod
    @memorize
    def calculateBits(field: int):
        return round(math.log(field, 2))

    def modulo(self, x):
        if self.dtype == "custom":
            mask_pos = x > self.max_value
            mask_neg = x < self.min_value
            if mask_pos.any():
                mask_pos = mask_pos.long()
                return self.modulo(x - (mask_pos * self.field))
            elif mask_neg.any():
                mask_neg = mask_neg.long()
                return self.modulo(x + (mask_neg * self.field))
            else:
                return x.type(self.torch_dtype)
        else:
            return x

    def get(self):
        """Fetches all shares and returns the plaintext tensor they represent"""

        shares = list()

        for share in self.child.values():
            if isinstance(share, sy.PointerTensor):
                shares.append(share.get())
            else:
                shares.append(share)
                self.owner.de_register_obj(share)

        # For dtype values long and int modulo is automatically handled by native torch tensors
        result = self.modulo(sum(shares))
        return result

    def virtual_get(self):
        """Get the value of the tensor without calling get
         - Useful for debugging, only for VirtualWorkers"""

        shares = list()

        for v in self.child.values():
            share = v.location._objects[v.id_at_location]
            shares.append(share)

        result = self.modulo(sum(shares))
        return result

    def init_shares(self, *owners):
        """Initializes shares and distributes them amongst their respective owners

        Args:
            *owners the list of shareholders. Can be of any length.

            """
        shares = self.generate_shares(
            self.child, n_workers=len(owners), random_type=self.torch_dtype
        )

        shares_dict = {}
        for share, owner in zip(shares, owners):
            share_ptr = share.send(owner, **no_wrap)
            shares_dict[share_ptr.location.id] = share_ptr

        self.child = shares_dict
        return self

    def generate_shares(self, secret, n_workers, random_type):
        """The cryptographic method for generating shares given a secret tensor.

        Args:
            secret: the tensor to be shared.
            n_workers: the number of shares to generate for each value
                (i.e., the number of tensors to return)
            random_type: the torch type shares should be encoded in (use the smallest possible)
                given the choice of mod"
            """
        random_type = torch.LongTensor if random_type == torch.int64 else torch.IntTensor
        if not isinstance(secret, random_type):
            secret = secret.type(random_type)

        random_shares = [random_type(secret.shape) for _ in range(n_workers - 1)]

        for share in random_shares:
            share.random_(self.min_value, self.max_value)
        shares = []
        for i in range(n_workers):
            if i == 0:
                share = random_shares[i]
            elif i < n_workers - 1:
                share = random_shares[i] - random_shares[i - 1]
            else:
                share = secret - random_shares[i - 1]
            shares.append(self.modulo(share))
        return shares

    def reconstruct(self):
        """
        Reconstruct the shares of the AdditiveSharingTensor remotely without
        its owner being able to see any sensitive value

        Returns:
            A MultiPointerTensor where all workers hold the reconstructed value
        """
        workers = self.locations

        ptr_to_sh = self.copy().wrap().send(workers[0], **no_wrap)
        pointer = ptr_to_sh.remote_get()

        pointers = [pointer]
        for worker in workers[1:]:
            pointers.append(pointer.copy().move(worker))

        return sy.MultiPointerTensor(children=pointers)

    def zero(self, shape=None):
        """
        Build an additive shared tensor of value zero with the same
        properties as self
        """

        if shape == None or len(shape) == 0:
            shape = self.shape if self.shape else [1]
        zero = torch.zeros(*shape, dtype=self.torch_dtype).share(
            *self.locations,
            field=self.field,
            dtype=self.dtype,
            crypto_provider=self.crypto_provider,
            **no_wrap,
        )
        return zero

    def refresh(self):
        """
        Refresh shares by adding shares of zero
        """
        zero = self.zero()
        r = self + zero
        return r

    @overloaded.overload_method
    def _getitem_multipointer(self, self_shares, indices_shares):
        """
        Support x[i] where x is an AdditiveSharingTensor and i a MultiPointerTensor

        Args:
            self_shares (dict): the dict of shares of x
            indices_shares (dict): the dict of shares of i

        Returns:
            an AdditiveSharingTensor
        """
        selected_shares = {}
        for worker, share in self_shares.items():
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

    @overloaded.overload_method
    def _getitem_public(self, self_shares, indices):
        """
        Support x[i] where x is an AdditiveSharingTensor and i a MultiPointerTensor

        Args:
            self_shares (dict): the dict of shares of x
            indices_shares (tuples of ints): integers indices

        Returns:
            an AdditiveSharingTensor

        """
        selected_shares = {}
        for worker, share in self_shares.items():
            selected_shares[worker] = share[indices]

        return selected_shares

    def __getitem__(self, indices):
        if not isinstance(indices, (tuple, list)):
            indices = (indices,)
        tensor_type = type(indices[-1])

        if tensor_type == sy.MultiPointerTensor:
            return self._getitem_multipointer(indices)
        else:
            return self._getitem_public(indices)

    ## SECTION SPDZ

    @overloaded.method
    def add(self, shares: dict, other):
        """Adds operand to the self AST instance.

        Args:
            shares: a dictionary <location_id -> PointerTensor) of shares corresponding to
                self. Equivalent to calling self.child.
            other: the operand being added to self, can be:
                - a dictionary <location_id -> PointerTensor) of shares
                - a torch tensor
                - a constant
        """
        if isinstance(other, int):
            other = torch.tensor([other], dtype=self.torch_dtype)

        if isinstance(other, (torch.LongTensor, torch.IntTensor)):
            # if someone passes a torch tensor, we share it and keep the dict
            other = other.share(
                *self.child.keys(),
                field=self.field,
                dtype=self.dtype,
                crypto_provider=self.crypto_provider,
                **no_wrap,
            ).child
        elif not isinstance(other, dict):
            # if someone passes in a constant, we cast it to a tensor, share it and keep the dict
            other = (
                torch.tensor([other], dtype=self.torch_dtype)
                .share(
                    *self.child.keys(),
                    field=self.field,
                    dtype=self.dtype,
                    crypto_provider=self.crypto_provider,
                    **no_wrap,
                )
                .child
            )

        assert len(shares) == len(other)

        # matches each share which needs to be added according
        # to the location of the share
        new_shares = {}
        for k, v in shares.items():
            new_shares[k] = self.modulo(other[k] + v)
        return new_shares

    __add__ = add
    __radd__ = add

    @overloaded.method
    def sub(self, shares: dict, other):
        """Subtracts an operand from the self AST instance.

        Args:
            shares: a dictionary <location_id -> PointerTensor) of shares corresponding to
                self. Equivalent to calling self.child.
            other: the operand being subtracted from self, can be:
                - a dictionary <location_id -> PointerTensor) of shares
                - a torch tensor
                - a constant
        """

        if isinstance(other, int):
            other = torch.tensor([other], dtype=self.torch_dtype)

        if isinstance(other, (torch.LongTensor, torch.IntTensor)):
            # if someone passes a torch tensor, we share it and keep the dict
            other = other.share(
                *self.child.keys(),
                field=self.field,
                dtype=self.dtype,
                crypto_provider=self.crypto_provider,
                **no_wrap,
            ).child
        elif not isinstance(other, dict):
            # if someone passes in a constant, we cast it to a tensor, share it and keep the dict
            other = (
                torch.tensor([other], dtype=self.torch_dtype)
                .share(
                    *self.child.keys(),
                    field=self.field,
                    dtype=self.dtype,
                    crypto_provider=self.crypto_provider,
                    **no_wrap,
                )
                .child
            )

        assert len(shares) == len(other)

        # matches each share which needs to be added according
        # to the location of the share
        new_shares = {}
        for k, v in shares.items():
            new_shares[k] = self.modulo(v - other[k])

        return new_shares

    __sub__ = sub

    def __rsub__(self, other):
        return (self - other) * -1

    def _private_mul(self, other, equation: str):
        """Abstractly Multiplies two tensors

        Args:
            self: an AdditiveSharingTensor
            other: another AdditiveSharingTensor
            equation: a string representation of the equation to be computed in einstein
                summation form
        """
        # check to see that operation is either mul or matmul
        assert equation == "mul" or equation == "matmul"
        cmd = getattr(torch, equation)

        assert isinstance(other, AdditiveSharingTensor)

        assert len(self.child) == len(other.child)

        if self.crypto_provider is None:
            raise AttributeError("For multiplication a crypto_provider must be passed.")

        shares = spdz.spdz_mul(cmd, self, other, self.crypto_provider, self.field, self.dtype)

        return shares

    @overloaded.method
    def _public_mul(self, shares, other, equation):
        """Multiplies an AdditiveSharingTensor with a non-private value
        (int, torch tensor, MultiPointerTensor, etc.)

        When other is a constant equal to zero, the shares vanish so we need to add fresh
        shares of zero.

        Args:
            shares (dict): a dictionary <location_id -> PointerTensor) of shares corresponding to
                self. Equivalent to calling self.child.
            other (dict of int): operand being multiplied with self, can be:
                - a dictionary <location_id -> PointerTensor) of shares
                - a torch tensor (Int or Long)
                - or an integer
            equation: a string representation of the equation to be computed in einstein
                summation form
        """
        assert equation == "mul" or equation == "matmul"
        cmd = getattr(torch, equation)
        if isinstance(other, dict):
            return {
                worker: (self.modulo(cmd(share, other[worker]))) for worker, share in shares.items()
            }
        else:
            other_is_zero = False
            if isinstance(other, (torch.LongTensor, torch.IntTensor)):
                if (other == 0).any():
                    other_is_zero = True
            elif other == 0:
                other_is_zero = True

            if other_is_zero:
                res = {}
                first_it = True

                for worker, share in shares.items():
                    cmd_res = cmd(share, other)
                    if first_it:
                        first_it = False
                        zero_shares = self.zero(cmd_res.shape).child
                    res[worker] = self.modulo(cmd(share, other) + zero_shares[worker])
                return res
            else:
                return {
                    worker: (self.modulo(cmd(share, other))) for worker, share in shares.items()
                }

    def mul(self, other):
        """Multiplies two tensors together

        Args:
            self (AdditiveSharingTensor): an AdditiveSharingTensor
            other: another AdditiveSharingTensor, or a MultiPointerTensor, or an integer
        """
        if not isinstance(other, sy.AdditiveSharingTensor):
            return self._public_mul(other, "mul")

        return self._private_mul(other, "mul")

    def __mul__(self, other, **kwargs):
        return self.mul(other, **kwargs)

    def __imul__(self, other):
        self = self.mul(other)
        return self

    def square(self):
        return self.mul(self)

    def pow(self, power):
        """
        Compute integer power of a number by recursion using mul

        This uses the following trick:
         - Divide power by 2 and multiply base to itself (if the power is even)
         - Decrement power by 1 to make it even and then follow the first step
        """
        base = self

        result = 1
        while power > 0:
            # If power is odd
            if power % 2 == 1:
                result = result * base

            # Divide the power by 2
            power = power // 2
            # Multiply base to itself
            base = base * base

        return result

    __pow__ = pow

    def matmul(self, other):
        """Multiplies two tensors matrices together

        Args:
            self: an AdditiveSharingTensor
            other: another AdditiveSharingTensor or a MultiPointerTensor
        """
        # If the multiplication can be public
        if not isinstance(other, sy.AdditiveSharingTensor):
            return self._public_mul(other, "matmul")

        return self._private_mul(other, "matmul")

    def mm(self, *args, **kwargs):
        """Multiplies two tensors matrices together
        """
        return self.matmul(*args, **kwargs)

    def __matmul__(self, *args, **kwargs):
        """Multiplies two tensors matrices together
        """
        return self.matmul(*args, **kwargs)

    def __itruediv__(self, *args, **kwargs):

        result = self.__truediv__(*args, **kwargs)
        self.child = result.child

    def _private_div(self, divisor):
        return securenn.division(self, divisor)

    @overloaded.method
    def _public_div(self, shares: dict, divisor):
        # TODO: how to correctly handle division in Zq?
        divided_shares = {}
        for i_worker, (location, pointer) in enumerate(shares.items()):
            # Still no solution to perform a real division on a additive shared tensor
            # without a heavy crypto protocol.
            # For now, the solution works in most cases when the tensor is shared between 2 workers
            divided_shares[location] = pointer / divisor

        return divided_shares

    def div(self, divisor):
        if isinstance(divisor, AdditiveSharingTensor):
            return self._private_div(divisor)
        else:
            return self._public_div(divisor)

    __truediv__ = div

    @overloaded.method
    def mod(self, shares: dict, modulus: int):
        assert isinstance(modulus, int)

        moded_shares = {}
        for location, pointer in shares.items():
            moded_shares[location] = pointer % modulus

        return moded_shares

    def __mod__(self, *args, **kwargs):
        return self.mod(*args, **kwargs)

    @overloaded.method
    def chunk(self, shares, *args, **kwargs):
        """
        This method overrides the torch.Tensor.chunk() method of Pytorch
        """
        results = None

        for worker, share in shares.items():
            share_results = share.chunk(*args, **kwargs)
            if isinstance(share_results, (tuple, list)):
                if results is None:
                    results = [{worker: share_result} for share_result in share_results]
                else:
                    for result, share_result in zip(results, share_results):
                        result[worker] = share_result
            else:
                if results is None:
                    results = {}
                results[worker] = share_results

        return results

    @staticmethod
    @overloaded.module
    def torch(module):
        def add(self, other):
            """Overload add(x, y) to redirect to add(y)"""
            return self.add(other)

        module.add = add

        def mul(self, other):
            """Overload torch.mul(x, y) to redirect to x.mul(y)"""
            return self.mul(other)

        module.mul = mul

        def matmul(self, other):
            """Overload torch.matmul(x, y) to redirect to x.matmul(y)"""
            return self.matmul(other)

        module.matmul = matmul

        def sum(self, *args, **kwargs):
            """Overload torch.sum(x) to redirect to x.sum()"""
            return self.sum(*args, **kwargs)

        module.sum = sum

        def dot(self, other):
            """Overload torch.dot(x, y)"""
            return self.mul(other).sum()

        module.dot = dot

        def mean(self, *args, **kwargs):
            """Overload torch.mean(x)"""
            # We cannot directly use mean on Long tensors
            # so we do it by hand with a sum and a division
            sum = self.sum(*args, **kwargs)

            # We need to know how many input values are used for each
            # output value to divide
            dims_to_reduce = args[0] if args else range(self.dim())
            if isinstance(dims_to_reduce, int):
                dims_to_reduce = (dims_to_reduce,)

            div = 1
            for i, s in enumerate(self.shape):
                if i in dims_to_reduce:
                    div *= s

            return sum // div

        module.mean = mean

        @overloaded.function
        def unbind(tensor_shares, **kwargs):
            results = None

            for worker, share in tensor_shares.items():
                share_results = torch.unbind(share, **kwargs)
                if results is None:
                    results = [{worker: share_result} for share_result in share_results]
                else:
                    for result, share_result in zip(results, share_results):
                        result[worker] = share_result

            return results

        module.unbind = unbind

        def share_combine(tensors_shares):
            results = {}
            workers = tensors_shares[0].keys()

            for worker in workers:
                tensors_share = []
                for tensor_shares in tensors_shares:
                    tensor_share = tensor_shares[worker]
                    tensors_share.append(tensor_share)
                results[worker] = tensors_share

            return results

        @overloaded.function
        def stack(tensors_shares, **kwargs):
            return {
                worker: torch.stack(share, **kwargs)
                for (worker, share) in share_combine(tensors_shares).items()
            }

        module.stack = stack

        @overloaded.function
        def cat(tensors_shares, **kwargs):
            return {
                worker: torch.cat(share, **kwargs)
                for (worker, share) in share_combine(tensors_shares).items()
            }

        module.cat = cat

        def chunk(tensor, *args, **kwargs):
            return tensor.chunk(*args, **kwargs)

        module.chunk = chunk

        @overloaded.function
        def roll(tensor_shares, shifts, **kwargs):
            """ Return a tensor where values are cyclically shifted compared to the original one.
            For instance, torch.roll([1, 2, 3], 1) returns torch.tensor([3, 1, 2]).
            In **kwargs should be dims, an argument to tell along which dimension the tensor should
            be rolled. If dims is None, the tensor is flattened, rolled, and restored to its original shape.
            shifts and dims can be tuples of same length to perform several rolls along different dimensions.
            """
            results = {}
            for worker, share in tensor_shares.items():
                if isinstance(shifts, dict):
                    results[worker] = torch.roll(share, shifts[worker], **kwargs)
                elif isinstance(shifts, tuple) and isinstance(shifts[0], dict):
                    worker_shifts = [s[worker] for s in shifts]
                    results[worker] = torch.roll(share, worker_shifts, **kwargs)
                else:
                    results[worker] = torch.roll(share, shifts, **kwargs)

            return results

        module.roll = roll

        def max(tensor, **kwargs):
            return tensor.max(**kwargs)

        module.max = max

        def argmax(tensor, **kwargs):
            return tensor.argmax(**kwargs)

        module.argmax = argmax

        @overloaded.module
        def functional(module):
            @overloaded.function
            def split(tensor_shares, *args, **kwargs):
                results = None

                for worker, share in tensor_shares.items():
                    share_results = torch.split(share, *args, **kwargs)
                    if results is None:
                        results = [{worker: share_result} for share_result in share_results]
                    else:
                        for result, share_result in zip(results, share_results):
                            result[worker] = share_result

                return results

            module.split = split

        module.functional = functional

        @overloaded.module
        def nn(module):
            @overloaded.module
            def functional(module):
                def relu(tensor_shares, inplace=False):
                    return tensor_shares.relu()

                module.relu = relu

                @overloaded.function
                def pad(input_shares, pad, mode="constant", value=0):
                    padded_shares = {}
                    for location, shares in input_shares.items():
                        padded_shares[location] = torch.nn.functional.pad(shares, pad, mode, value)

                    return padded_shares

                module.pad = pad

            module.functional = functional

        module.nn = nn

    ## SECTION SNN

    def relu(self, inplace=False):
        return securenn.relu(self)

    def positive(self):
        # self >= 0
        return securenn.relu_deriv(self)

    def gt(self, other):
        r = self - other - 1
        return r.positive()

    @crypto_protocol("snn")
    def __gt__(self, other):
        return self.gt(other)

    @crypto_protocol("fss")
    def __gt__(self, other):
        return (other + 1) <= self

    def ge(self, other):
        return (self - other).positive()

    @crypto_protocol("snn")
    def __ge__(self, other):
        return self.ge(other)

    @crypto_protocol("fss")
    def __ge__(self, other):
        return other <= self

    def lt(self, other):
        return (other - self - 1).positive()

    @crypto_protocol("snn")
    def __lt__(self, other):
        return self.lt(other)

    @crypto_protocol("fss")
    def __lt__(self, other):
        return (self + 1) <= other

    def le(self, other):
        return (other - self).positive()

    @crypto_protocol("snn")
    def __le__(self, other):
        return self.le(other)

    @crypto_protocol("fss")
    def __le__(self, other):
        return fss.le(self, other)

    @crypto_protocol("snn")
    def eq(self, other):
        diff = self - other
        diff2 = diff * diff
        negdiff2 = diff2 * -1
        return negdiff2.positive()

    @crypto_protocol("fss")
    def eq(self, other):
        return fss.eq(self, other)

    def __eq__(self, other):
        return self.eq(other)

    def max(self, dim=None, return_idx=False):
        """
        Return the maximum value of an additive shared tensor

        Args:
            dim (None or int): if not None, the dimension on which
                the comparison should be done
            return_idx (bool): Return the index of the maximum value
                Note that if dim is specified then the index is returned
                anyway to match the Pytorch syntax.

        return:
            the maximum value (possibly across an axis)
            and optionally the index of the maximum value (possibly across an axis)
        """
        values = self
        n_dim = self.dim()

        # Make checks and transformation
        assert dim is None or (0 <= dim < n_dim), f"Dim overflow  0 <= {dim} < {n_dim}"
        # FIXME make it cleaner and robust for more options
        if n_dim == 2:
            if dim == None:
                values = values.view(-1)
            elif dim == 1:
                values = values.t()
        assert n_dim <= 2, "Max on tensor with len(shape) > 2 is not supported."

        # Init max vals and idx to the first element
        max_value = values[0]
        max_index = torch.tensor([0]).share(
            *self.locations,
            field=self.field,
            dtype=self.dtype,
            crypto_provider=self.crypto_provider,
            **no_wrap,
        )

        for i in range(1, len(values)):
            a = values[i]
            beta = a >= max_value
            max_index = max_index + beta * (i - max_index)
            max_value = max_value + beta * (a - max_value)

        if dim is None and return_idx is False:
            return max_value
        else:
            return max_value, max_index * 1000

    def argmax(self, dim=None):

        max_value, max_index = self.max(dim=dim, return_idx=True)

        return max_index

    ## STANDARD

    @staticmethod
    def select_worker(args_, worker):
        """
        utility function for handle_func_command which help to select
        shares (seen as elements of dict) in an argument set. It could
        perhaps be put elsewhere

        Args:
            args_: arguments to give to a functions
            worker: owner of the shares to select

        Return:
            args_ where the AdditiveSharedTensors are replaced by
            the appropriate share
        """
        return map(lambda x: x[worker] if isinstance(x, dict) else x, args_)

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
            <no self>, arguments[, kwargs_])

        Returns:
            the response of the function command
        """
        cmd, _, args_, kwargs_ = command

        # Check that the function has not been overwritten
        try:
            # Try to get recursively the attributes in cmd = "<attr1>.<attr2>.<attr3>..."
            cmd = cls.rgetattr(cls, cmd)
        except AttributeError:
            pass
        if not isinstance(cmd, str):
            return cmd(*args_, **kwargs_)

        tensor = args_[0] if not isinstance(args_[0], (tuple, list)) else args_[0][0]

        # Replace all SyftTensors with their child attribute
        new_args, new_kwargs, new_type = hook_args.unwrap_args_from_function(cmd, args_, kwargs_)

        results = {}
        for worker, share in new_args[0].items():
            new_type = type(share)
            new_args_worker = tuple(AdditiveSharingTensor.select_worker(new_args, worker))

            # build the new command
            new_command = (cmd, None, new_args_worker, new_kwargs)

            # Send it to the appropriate class and get the response
            results[worker] = new_type.handle_func_command(new_command)

        # Put back AdditiveSharingTensor on the tensors found in the response
        response = hook_args.hook_response(
            cmd, results, wrap_type=cls, wrap_args=tensor.get_class_attributes()
        )

        return response

    def set_garbage_collect_data(self, value):
        shares = self.child
        for _, share in shares.items():
            share.garbage_collect_data = value

    def get_garbage_collect_data(self):
        garbage_collect_data_dict = dict()
        shares = self.child

        for worker, share in shares.items():
            garbage_collect_data_dict[worker] = share.garbage_collect_data

        return garbage_collect_data_dict

    @staticmethod
    def simplify(worker: AbstractWorker, tensor: "AdditiveSharingTensor") -> tuple:
        """
        This function takes the attributes of a AdditiveSharingTensor and saves them in a tuple
        Args:
            tensor (AdditiveSharingTensor): a AdditiveSharingTensor
        Returns:
            tuple: a tuple holding the unique attributes of the additive shared tensor
        Examples:
            data = simplify(tensor)
        """

        chain = None
        if hasattr(tensor, "child"):
            chain = sy.serde.msgpack.serde._simplify(worker, tensor.child)

        # Don't delete the remote values of the shares at simplification
        garbage_collect = tensor.get_garbage_collect_data()
        tensor.set_garbage_collect_data(False)

        return (
            sy.serde.msgpack.serde._simplify(worker, tensor.id),
            sy.serde.msgpack.serde._simplify(worker, tensor.field),
            tensor.dtype.encode("utf-8"),
            sy.serde.msgpack.serde._simplify(worker, tensor.crypto_provider.id),
            chain,
            garbage_collect,
        )

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "AdditiveSharingTensor":
        """
            This function reconstructs a AdditiveSharingTensor given it's attributes in form of a tuple.
            Args:
                worker: the worker doing the deserialization
                tensor_tuple: a tuple holding the attributes of the AdditiveSharingTensor
            Returns:
                AdditiveSharingTensor: a AdditiveSharingTensor
            Examples:
                shared_tensor = detail(data)
            """
        tensor_id, field, dtype, crypto_provider, chain, garbage_collect = tensor_tuple

        crypto_provider = sy.serde.msgpack.serde._detail(worker, crypto_provider)

        tensor = AdditiveSharingTensor(
            owner=worker,
            id=sy.serde.msgpack.serde._detail(worker, tensor_id),
            field=sy.serde.msgpack.serde._detail(worker, field),
            dtype=dtype.decode("utf-8"),
            crypto_provider=worker.get_worker(crypto_provider),
        )

        if chain is not None:
            chain = sy.serde.msgpack.serde._detail(worker, chain)
            tensor.child = chain

        tensor.set_garbage_collect_data(garbage_collect)

        return tensor

    @staticmethod
    def bufferize(
        worker: AbstractWorker, tensor: "AdditiveSharingTensor"
    ) -> "AdditiveSharingTensorPB":
        """
        This function takes the attributes of a AdditiveSharingTensor and saves them in a protobuf object
        Args:
            tensor (AdditiveSharingTensor): a AdditiveSharingTensor
        Returns:
            protobuf: a protobuf object holding the unique attributes of the additive shared tensor
        Examples:
            data = protobuf(tensor)
        """
        protobuf_tensor = AdditiveSharingTensorPB()

        if hasattr(tensor, "child"):
            for key, value in tensor.child.items():
                sy.serde.protobuf.proto.set_protobuf_id(protobuf_tensor.location_ids.add(), key)
                protobuf_share = sy.serde.protobuf.serde._bufferize(worker, value)
                protobuf_tensor.shares.append(protobuf_share)

        # Don't delete the remote values of the shares at simplification
        tensor.set_garbage_collect_data(False)

        sy.serde.protobuf.proto.set_protobuf_id(protobuf_tensor.id, tensor.id)
        sy.serde.protobuf.proto.set_protobuf_id(
            protobuf_tensor.crypto_provider_id, tensor.crypto_provider.id
        )

        if tensor.field >= 2 ** 64:
            protobuf_tensor.field_str = str(tensor.field)
        else:
            protobuf_tensor.field_int = tensor.field
        protobuf_tensor.dtype = tensor.dtype

        return protobuf_tensor

    @staticmethod
    def unbufferize(
        worker: AbstractWorker, protobuf_tensor: "AdditiveSharingTensorPB"
    ) -> "AdditiveSharingTensor":
        """
            This function reconstructs a AdditiveSharingTensor given its' attributes in form of a protobuf object.
            Args:
                worker: the worker doing the deserialization
                protobuf_tensor: a protobuf object holding the attributes of the AdditiveSharingTensor
            Returns:
                AdditiveSharingTensor: a AdditiveSharingTensor
            Examples:
                shared_tensor = unprotobuf(data)
            """

        tensor_id = sy.serde.protobuf.proto.get_protobuf_id(protobuf_tensor.id)
        crypto_provider_id = sy.serde.protobuf.proto.get_protobuf_id(
            protobuf_tensor.crypto_provider_id
        )
        field = int(getattr(protobuf_tensor, protobuf_tensor.WhichOneof("field_size")))
        dtype = protobuf_tensor.dtype

        tensor = AdditiveSharingTensor(
            owner=worker,
            id=tensor_id,
            field=field,
            dtype=dtype,
            crypto_provider=worker.get_worker(crypto_provider_id),
        )

        if protobuf_tensor.location_ids is not None:
            chain = {}
            for pb_location_id, share in zip(protobuf_tensor.location_ids, protobuf_tensor.shares):
                location_id = sy.serde.protobuf.proto.get_protobuf_id(pb_location_id)
                chain[location_id] = sy.serde.protobuf.serde._unbufferize(worker, share)
            tensor.child = chain

        return tensor


### Register the tensor with hook_args.py ###
hook_args.default_register_tensor(AdditiveSharingTensor)
