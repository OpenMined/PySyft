import math
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
            id: An optional string or integer id of the LoggingTensor.
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

        for share in self.child.values():
            if hasattr(share, "child") and isinstance(share.child, sy.PointerTensor):
                shares.append(share.get())
            else:
                shares.append(share.child)
        return sum(shares)

    def virtual_get(self):
        """Get the value of the tensor without calling get
         - Useful for debugging, only for VirtualWorkers"""

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

    def reconstruct(self):
        """
        Reconstruct the shares of the AdditiveSharingTensor remotely without
        its owner being able to see any sensitive value

        Returns:
            A MultiPointerTensor where all workers hold the reconstructed value
        """
        workers = self.locations

        ptr_to_sh = self.wrap().send(workers[0])
        pointer = ptr_to_sh.remote_get()

        pointers = [pointer]
        for worker in workers[1:]:
            pointers.append(pointer.send(worker).remote_get())

        return sy.MultiPointerTensor(children=pointers)

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
        if not isinstance(indices, tuple):
            indices = (indices,)
        tensor_type = type(indices[-1])
        if tensor_type == sy.MultiPointerTensor:
            return self._getitem_multipointer(indices)
        else:
            return self._getitem_public(indices)

    ## SECTION SPDZ

    @overloaded.method
    def add(self, shares: dict, other_shares):
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
    def sub(self, shares: dict, other_shares):
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
            raise AttributeError("For multiplication a cryto_provider must be passed.")

        shares = spdz.spdz_mul(cmd, self, other, self.crypto_provider, self.field)

        return shares

    @overloaded.method
    def _public_mul(self, shares, other, equation):
        """Multiplies an AdditiveSharingTensor with a non-private value
        (int, MultiPointerTensor, etc.)

        Args:
            shares (dict): a dictionary <location_id -> PointerTensor) of shares corresponding to
                self. Equivalent to calling self.child.
            other (dict of int): a dictionary <location_id -> PointerTensor) of shares corresponding
                to the tensor being multiplied with self or an integer
            equation: a string representation of the equation to be computed in einstein
                summation form
        """
        assert equation == "mul" or equation == "matmul"
        cmd = getattr(torch, equation)

        if isinstance(other, dict):
            return {worker: cmd(share, other[worker]) for worker, share in shares.items()}
        else:
            return {worker: cmd(share, other) for worker, share in shares.items()}

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

    @staticmethod
    @overloaded.module
    def torch(module):
        def mul(self, other):
            """Overload torch.mul(x, y) to redirect to x.mul(y)"""
            return self.mul(other)

        module.mul = mul

        def matmul(self, other):
            """Overload torch.matmul(x, y) to redirect to x.matmul(y)"""
            return self.matmul(other)

        module.matmul = matmul

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

        def max(tensor, **kwargs):
            return tensor.max(**kwargs)

        module.max = max

        def argmax(tensor, **kwargs):
            return tensor.argmax(**kwargs)

        module.argmax = argmax

        def conv2d(
            input,
            weight,
            bias=None,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            padding_mode="zeros",
        ):
            """
            Overloads torch.conv2d to be able to use MPC on convolutional networks.
            The idea is to build new tensors from input and weight to compute a matrix multiplication
            equivalent to the convolution.

            Args:
                input: input image
                weight: convolution kernels
                bias: optional additive bias
                stride: stride of the convolution kernels
                padding:
                dilation: spacing between kernel elements
                groups:
                padding_mode: type of padding, should be either 'zeros' or 'circular' but 'reflect' and 'replicate' accepted
            Returns:
                the result of the convolution as an AdditiveSharingTensor
            """
            assert len(input.shape) == 4
            assert len(weight.shape) == 4

            # We could make a util function for these, as PyTorch's _pair
            if isinstance(stride, int):
                stride = (stride, stride)
            if isinstance(padding, int):
                padding = (padding, padding)
            if isinstance(dilation, int):
                dilation = (dilation, dilation)

            # Extract a few useful values
            batch_size, nb_channels_in, nb_rows_in, nb_cols_in = input.shape
            nb_channels_out, nb_channels_in_, nb_rows_kernel, nb_cols_kernel = weight.shape

            if bias is not None:
                assert len(bias) == nb_channels_out

            # Check if inputs are coherent
            assert nb_channels_in == nb_channels_in_ * groups
            assert nb_channels_in % groups == 0
            assert nb_channels_out % groups == 0

            # Compute output shape
            nb_rows_out = int(
                ((nb_rows_in + 2 * padding[0] - dilation[0] * (nb_rows_kernel - 1) - 1) / stride[0])
                + 1
            )
            nb_cols_out = int(
                ((nb_cols_in + 2 * padding[1] - dilation[1] * (nb_cols_kernel - 1) - 1) / stride[1])
                + 1
            )

            # Apply padding to the input
            if padding != (0, 0):
                padding_mode = "constant" if padding_mode == "zeros" else padding_mode
                input = torch.nn.functional.pad(
                    input, (padding[1], padding[1], padding[0], padding[0]), padding_mode
                )
                # Update shape after padding
                nb_rows_in += 2 * padding[0]
                nb_cols_in += 2 * padding[1]

            # We want to get relative positions of values in the input tensor that are used by one filter convolution.
            pattern_ind = []
            for ch in range(nb_channels_in):
                for r in range(nb_rows_kernel):
                    for c in range(nb_cols_kernel):
                        pixel = r * nb_cols_in * dilation[0] + (c % nb_cols_kernel) * dilation[1]
                        pattern_ind.extend([pixel + ch * nb_rows_in * nb_cols_in])

            # The image tensor is reshaped for the matrix multiplication:
            # on each row of the new tensor will be the input values used for each filter convolution
            im_flat = input.view(batch_size, -1)
            im_reshaped = []
            for cur_row_out in range(nb_rows_out):
                for cur_col_out in range(nb_cols_out):
                    offset = (
                        cur_row_out * stride[0] * nb_cols_in
                        + (cur_col_out % nb_cols_out) * stride[1]
                    )
                    tmp = [ind + offset for ind in pattern_ind]
                    im_reshaped.append(im_flat[:, tmp].wrap())
            im_reshaped = torch.stack(im_reshaped).permute(1, 0, 2)

            # The convolution kernels are also reshaped for the matrix multiplication
            weight_reshaped = weight.view(nb_channels_out // groups, -1).t().wrap()

            # Now that everything is set up, we can compute the result
            if groups > 1:
                res = []
                chunks_im = torch.chunk(im_reshaped, groups, dim=2)
                chunks_weights = torch.chunk(weight_reshaped, groups, dim=0)
                for g in range(groups):
                    tmp = chunks_im[g].matmul(chunks_weights[g])
                    res.append(tmp)
                res = torch.cat(res, dim=2).child
            else:
                res = im_reshaped.matmul(weight_reshaped).child

            # Add a bias if needed
            if bias is not None:
                if bias.is_wrapper:
                    res = res + bias.child  # += does not work
                else:
                    res = res + bias

            # ... And reshape it back to an image
            res = (
                res.permute(0, 2, 1)
                .view(batch_size, nb_channels_out, nb_rows_out, nb_cols_out)
                .contiguous()
            )
            return res

        module.conv2d = conv2d

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

                def pad(input, pad, mode="constant", value=0):
                    padded_shares = {}
                    shares_dict = input.child.child if input.is_wrapper else input.child
                    for location, shares in shares_dict.items():
                        padded_shares[location] = torch.nn.functional.pad(shares, pad, mode, value)

                    return AdditiveSharingTensor(padded_shares)

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
                Note the if dim is specified then the index is returned
                anyway to match the Pytorch syntax.

        return:
            the maximum value (possibly across an axis)
            and optionally the index of the maximum value (possibly across an axis)
        """
        values = self
        n_dim = len(self.shape)

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
        max_index = (
            torch.tensor([0])
            .share(*self.locations, field=self.field, crypto_provider=self.crypto_provider)
            .child
        )

        for i in range(1, len(values)):
            a = values[i]
            beta = a >= max_value
            max_index = i * beta - max_index * (beta - 1)
            max_value = a * beta - max_value * (beta - 1)

        if dim is None and return_idx is False:
            return max_value
        else:
            return max_value, max_index * 1000

    def argmax(self, dim=None):

        max_value, max_index = self.max(dim=dim, return_idx=True)

        return max_index

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

        tensor = args[0] if not isinstance(args[0], tuple) else args[0][0]

        # Check that the function has not been overwritten
        try:
            # Try to get recursively the attributes in cmd = "<attr1>.<attr2>.<attr3>..."
            cmd = cls.rgetattr(cls, cmd)
        except AttributeError:
            pass
        if not isinstance(cmd, str):
            return cmd(*args, **kwargs)

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

    def set_garbage_collect_data(self, value):
        shares = self.child
        for _, share in shares.items():
            share.child.garbage_collect_data = value
