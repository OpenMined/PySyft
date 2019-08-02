import math
import torch
import syft as sy
from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
from syft.frameworks.torch.overload_torch import overloaded

from syft.workers import AbstractWorker

# Crypto protocols
from syft.frameworks.torch.crypto import spdz
from syft.frameworks.torch.crypto import securenn

no_wrap = {"no_wrap": True}


class AdditiveSharingTensor(AbstractTensor):
    def __init__(
        self,
        shares: dict = None,
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

            shares: Optional dictionary with the shares already split
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: An optional string or integer id of the AdditiveSharingTensor.
            field: size of the arithmetic field in which the shares live
            n_bits: linked to the field with the relation (2 ** nbits) == field
            crypto_provider: an optional BaseWorker providing crypto elements
                such as Beaver triples
            tags: an optional set of hashtags corresponding to this tensor
                which this tensor should be searchable for
            description: an optional string describing the purpose of the
                tensor
        """
        super().__init__(id=id, owner=owner, tags=tags, description=description)

        self.child = shares

        self.field = (2 ** securenn.Q_BITS) if field is None else field  # < 63 bits
        self.n_bits = (
            n_bits if n_bits is not None else max(8, round(math.log(self.field, 2)))
        )  # < 63 bits
        # assert 2 ** self.n_bits == self.field
        self.crypto_provider = (
            crypto_provider if crypto_provider is not None else sy.hook.local_worker
        )

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

    def dim(self):
        for share in self.child.values():
            return len(share.shape)

    def get_class_attributes(self):
        """
        Specify all the attributes need to build a wrapper correctly when returning a response,
        for example precision_fractional is important when wrapping the result of a method
        on a self which is a fixed precision tensor with a non default precision_fractional.
        """
        return {"crypto_provider": self.crypto_provider, "field": self.field, "n_bits": self.n_bits}

    @property
    def grad(self):
        """
        Gradient makes no sense for Additive Shared Tensor, so we make it clear
        that if someone query .grad on a Additive Shared Tensor it doesn't error
        but returns grad and can't be set
        """
        return None

    def get(self):
        """Fetches all shares and returns the plaintext tensor they represent"""

        shares = list()

        for share in self.child.values():
            if isinstance(share, sy.PointerTensor):
                shares.append(share.get())
            else:
                shares.append(share)

        res_field = sum(shares) % self.field

        gate = res_field.native_gt(self.field / 2).long()
        neg_nums = (res_field - self.field) * gate
        pos_nums = res_field * (1 - gate)
        result = neg_nums + pos_nums

        return result

    def virtual_get(self):
        """Get the value of the tensor without calling get
         - Useful for debugging, only for VirtualWorkers"""

        shares = list()

        for v in self.child.values():
            share = v.location._objects[v.id_at_location]
            shares.append(share)

        res_field = sum(shares) % self.field

        gate = res_field.native_gt(self.field / 2).long()
        neg_nums = (res_field - self.field) * gate
        pos_nums = res_field * (1 - gate)
        result = neg_nums + pos_nums

        return result

    def init_shares(self, *owners):
        """Initializes shares and distributes them amongst their respective owners

        Args:
            *owners the list of shareholders. Can be of any length.

            """
        shares = self.generate_shares(
            self.child, n_workers=len(owners), field=self.field, random_type=torch.LongTensor
        )

        shares_dict = {}
        for share, owner in zip(shares, owners):
            share_ptr = share.send(owner, **no_wrap)
            shares_dict[share_ptr.location.id] = share_ptr

        self.child = shares_dict
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
            share %= field  # Generated shares should be in a finite field Zq
            shares.append(share)

        return shares

    def reconstruct(self):
        """
        Reconstruct the shares of the AdditiveSharingTensor remotely without
        its owner being able to see any sensitive value

        Returns:
            A MultiPointerTensor where all workers hold the reconstructed value
        """
        workers = self.locations

        ptr_to_sh = self.wrap().send(workers[0], **no_wrap)
        pointer = ptr_to_sh.remote_get()

        pointers = [pointer]
        for worker in workers[1:]:
            pointers.append(pointer.copy().move(worker))

        return sy.MultiPointerTensor(children=pointers)

    def zero(self):
        """
        Build an additive shared tensor of value zero with the same
        properties as self
        """
        shape = self.shape if self.shape else [1]
        zero = (
            torch.zeros(*shape)
            .long()
            .share(
                *self.locations, field=self.field, crypto_provider=self.crypto_provider, **no_wrap
            )
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
        if isinstance(other, (torch.LongTensor, torch.IntTensor)):
            # if someone passes a torch tensor, we share it and keep the dict
            other = other.share(
                *self.child.keys(),
                field=self.field,
                crypto_provider=self.crypto_provider,
                **no_wrap,
            ).child
        elif not isinstance(other, dict):
            # if someone passes in a constant, we cast it to a tensor, share it and keep the dict
            other = (
                torch.Tensor([other])
                .share(
                    *self.child.keys(),
                    field=self.field,
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
            new_shares[k] = (other[k] + v) % self.field

        return new_shares

    def __add__(self, other, **kwargs):
        """Adds two tensors. Forwards command to add. See add() for more details."""

        return self.add(other, **kwargs)

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

        if isinstance(other, (torch.LongTensor, torch.IntTensor)):
            # if someone passes a torch tensor, we share it and keep the dict
            other = other.share(
                *self.child.keys(),
                field=self.field,
                crypto_provider=self.crypto_provider,
                **no_wrap,
            ).child
        elif not isinstance(other, dict):
            # if someone passes in a constant, we cast it to a tensor, share it and keep the dict
            other = (
                torch.tensor([other])
                .share(
                    *self.child.keys(),
                    field=self.field,
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
            new_shares[k] = (v - other[k]) % self.field

        return new_shares

    def __sub__(self, *args, **kwargs):
        """Subtracts two tensors. Forwards command to sub. See .sub() for details."""
        return self.sub(*args, **kwargs)

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

        shares = spdz.spdz_mul(cmd, self, other, self.crypto_provider, self.field)

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
                worker: (cmd(share, other[worker]) % self.field) for worker, share in shares.items()
            }
        else:
            other_is_zero = False
            if isinstance(other, (torch.LongTensor, torch.IntTensor)):
                if (other == 0).any():
                    other_is_zero = True
            elif other == 0:
                other_is_zero = True

            if other_is_zero:
                zero_shares = self.zero().child
                return {
                    worker: ((cmd(share, other) + zero_shares[worker]) % self.field)
                    for worker, share in shares.items()
                }
            else:
                return {
                    worker: (cmd(share, other) % self.field) for worker, share in shares.items()
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
        """Multiplies two number for details see mul
        """
        return self.mul(other, **kwargs)

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

    @overloaded.method
    def __truediv__(self, shares: dict, divisor):
        # TODO: how to correctly handle division in Zq?
        assert isinstance(divisor, int)

        divided_shares = {}
        for i_worker, (location, pointer) in enumerate(shares.items()):
            # Still no solution to perform a real division on a additive shared tensor
            # without a heavy crypto protocol.
            # For now, the solution works in most cases when the tensor is shared between 2 workers
            # The idea is to compute Q - (Q - pointer) / divisor for as many worker
            # as the number of times the sum of shares "crosses" Q/2.
            if i_worker % 2 == 0:
                divided_shares[location] = self.field - (self.field - pointer) / divisor
            else:
                divided_shares[location] = pointer / divisor

        return divided_shares

    div = __truediv__

    @overloaded.method
    def mod(self, shares: dict, modulus: int):
        assert isinstance(modulus, int)

        moded_shares = {}
        for location, pointer in shares.items():
            moded_shares[location] = pointer % modulus

        return moded_shares

    def __mod__(self, *args, **kwargs):
        return self.mod(*args, **kwargs)

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

        @overloaded.function
        def stack(tensors_shares, **kwargs):

            results = {}

            workers = tensors_shares[0].keys()

            for worker in workers:
                tensors_share = []
                for tensor_shares in tensors_shares:
                    tensor_share = tensor_shares[worker]
                    tensors_share.append(tensor_share)
                stacked_share = torch.stack(tensors_share, **kwargs)
                results[worker] = stacked_share

            return results

        module.stack = stack

        @overloaded.function
        def cat(tensors_shares, **kwargs):
            # The code is the same for cat and stack, maybe we could factorize

            results = {}

            workers = tensors_shares[0].keys()

            for worker in workers:
                cat_share = []
                for tensor_shares in tensors_shares:
                    tensor_share = tensor_shares[worker]
                    cat_share.append(tensor_share)
                results[worker] = torch.cat(cat_share, **kwargs)

            return results

        module.cat = cat

        @overloaded.function
        def chunk(tensor_shares, *args, **kwargs):
            worker_chunks = {}
            results = []

            for worker, share in tensor_shares.items():
                chunked_share = torch.chunk(share, *args, **kwargs)
                worker_chunks[worker] = chunked_share

            for c in range(len(worker_chunks[worker])):
                shared_chunk = {}
                for worker in tensor_shares.keys():
                    shared_chunk[worker] = worker_chunks[worker][c]

                results.append(shared_chunk)

            return results

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
                def relu(tensor_shares):
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
            *self.locations, field=self.field, crypto_provider=self.crypto_provider, **no_wrap
        )

        for i in range(1, len(values)):
            a = values[i]
            beta = a >= max_value
            max_index = max_index + beta * (-max_index + i)  # TODO i - max_index doesn't work
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
    def select_worker(args, worker):
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

        # Check that the function has not been overwritten
        try:
            # Try to get recursively the attributes in cmd = "<attr1>.<attr2>.<attr3>..."
            cmd = cls.rgetattr(cls, cmd)
        except AttributeError:
            pass
        if not isinstance(cmd, str):
            return cmd(*args, **kwargs)

        tensor = args[0] if not isinstance(args[0], tuple) else args[0][0]

        # TODO: I can't manage the import issue, can you?
        # Replace all SyftTensors with their child attribute
        new_args, new_kwargs, new_type = sy.frameworks.torch.hook_args.unwrap_args_from_function(
            cmd, args, kwargs
        )

        results = {}
        for worker, share in new_args[0].items():
            new_type = type(share)
            new_args_worker = tuple(AdditiveSharingTensor.select_worker(new_args, worker))

            # build the new command
            new_command = (cmd, None, new_args_worker, new_kwargs)

            # Send it to the appropriate class and get the response
            results[worker] = new_type.handle_func_command(new_command)

        # Put back AdditiveSharingTensor on the tensors found in the response
        response = sy.frameworks.torch.hook_args.hook_response(
            cmd, results, wrap_type=cls, wrap_args=tensor.get_class_attributes()
        )

        return response

    def set_garbage_collect_data(self, value):
        shares = self.child
        for _, share in shares.items():
            share.garbage_collect_data = value

    @staticmethod
    def simplify(tensor: "AdditiveSharingTensor") -> tuple:
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
            chain = sy.serde._simplify(tensor.child)

        # Don't delete the remote values of the shares at simplification
        tensor.set_garbage_collect_data(False)

        return (tensor.id, tensor.field, tensor.crypto_provider.id, chain)

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

        tensor_id, field, crypto_provider, chain = tensor_tuple

        tensor = AdditiveSharingTensor(
            owner=worker,
            id=tensor_id,
            field=field,
            crypto_provider=worker.get_worker(crypto_provider),
        )

        if chain is not None:
            chain = sy.serde._detail(worker, chain)
            tensor.child = chain

        return tensor
