import torch

import syft
from syft.workers import AbstractWorker
from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
from syft.frameworks.torch.tensors.interpreters.additive_shared import AdditiveSharingTensor
from syft.frameworks.torch.tensors.interpreters.multi_pointer import MultiPointerTensor
from syft.frameworks.torch.overload_torch import overloaded


class FixedPrecisionTensor(AbstractTensor):
    def __init__(
        self,
        owner=None,
        id=None,
        field: int = 2 ** 62,
        base: int = 10,
        precision_fractional: int = 3,
        kappa: int = 1,
        tags: set = None,
        description: str = None,
    ):
        """Initializes a Fixed Precision tensor, which encodes all decimal point
        values using an underlying integer value.

        The FixedPrecision enables to manipulate floats over an interface which
        supports only integers, Such as _SPDZTensor.

        This is done by specifying a precision p and given a float x,
        multiply it with 10**p before rounding to an integer (hence you keep
        p decimals)

        Args:
            owner: An optional BaseWorker object to specify the worker on which
                the tensor is located.
            id: An optional string or integer id of the FixedPrecisionTensor.
        """
        super().__init__(tags, description)

        self.owner = owner
        self.id = id
        self.child = None

        self.field = field
        self.base = base
        self.precision_fractional = precision_fractional
        self.kappa = kappa
        self.torch_max_value = torch.tensor(self.field).long()

    def get_class_attributes(self):
        """
        Specify all the attributes need to build a wrapper correctly when returning a response,
        for example precision_fractional is important when wrapping the result of a method
        on a self which is a fixed precision tensor with a non default precision_fractional.
        """
        return {
            "field": self.field,
            "base": self.base,
            "precision_fractional": self.precision_fractional,
            "kappa": self.kappa,
        }

    @property
    def data(self):
        return self

    @data.setter
    def data(self, new_data):
        self.child = new_data.child
        return self

    @property
    def grad(self):
        """
        Gradient makes no sense for Fixed Precision Tensor, so we make it clear
        that if someone query .grad on a Fixed Precision Tensor it doesn't error
        but returns grad and can't be set
        """
        return None

    def attr(self, attr_name):
        return self.__getattribute__(attr_name)

    def fix_precision(self, check_range=True):
        """This method encodes the .child object using fixed precision"""

        rational = self.child

        upscaled = (rational * self.base ** self.precision_fractional).long()

        if check_range:
            assert (
                upscaled.abs() < (self.field / 2)
            ).all(), (
                f"{rational} cannot be correctly embedded: choose bigger field or a lower precision"
            )

        field_element = upscaled % self.field
        field_element.owner = rational.owner

        self.child = field_element
        return self

    def float_precision(self):
        """this method returns a new tensor which has the same values as this
        one, encoded with floating point precision"""

        value = self.child.long() % self.field

        gate = value.native_gt(self.torch_max_value / 2).long()
        neg_nums = (value - self.field) * gate
        pos_nums = value * (1 - gate)
        result = (neg_nums + pos_nums).float() / (self.base ** self.precision_fractional)

        return result

    def truncate(self, precision_fractional, check_sign=True):
        truncation = self.base ** precision_fractional

        # We need to make sure that values are truncated "towards 0"
        # i.e. for a field of 100, 70 (equivalent to -30), should be truncated
        # at 97 (equivalent to -3), not 7
        if isinstance(self.child, AdditiveSharingTensor) or not check_sign:  # Handle FPT>(wrap)>AST
            self.child = self.child / truncation
            return self
        else:
            gate = self.child.native_gt(self.torch_max_value / 2).long()
            neg_nums = (self.child - self.field) / truncation + self.field
            pos_nums = self.child / truncation
            self.child = neg_nums * gate + pos_nums * (1 - gate)
            return self

    @overloaded.method
    def add(self, _self, other):
        """Add two fixed precision tensors together.
        """
        if isinstance(other, int):
            scaled_int = other * self.base ** self.precision_fractional
            return getattr(_self, "add")(scaled_int)

        if isinstance(_self, AdditiveSharingTensor) and isinstance(other, torch.Tensor):
            # If we try to add a FPT>(wrap)>AST and a FPT>torch.tensor,
            # we want to perform AST + torch.tensor
            other = other.wrap()
        elif isinstance(other, AdditiveSharingTensor) and isinstance(_self, torch.Tensor):
            # If we try to add a FPT>torch.tensor and a FPT>(wrap)>AST,
            # we swap operators so that we do the same operation as above
            _self, other = other, _self.wrap()

        response = getattr(_self, "add")(other)
        response %= self.field  # Wrap around the field

        return response

    __add__ = add

    def add_(self, value_or_tensor, tensor=None):
        if tensor is None:
            result = self.add(value_or_tensor)
        else:
            result = self.add(value_or_tensor * tensor)

        self.child = result.child
        return self

    def __iadd__(self, other):
        """Add two fixed precision tensors together.
        """
        self.child = self.add(other).child

        return self

    @overloaded.method
    def sub(self, _self, other):
        """Subtracts a fixed precision tensor from another one.
        """
        if isinstance(other, int):
            scaled_int = other * self.base ** self.precision_fractional
            return getattr(_self, "sub")(scaled_int)

        if isinstance(_self, AdditiveSharingTensor) and isinstance(other, torch.Tensor):
            # If we try to subtract a FPT>(wrap)>AST and a FPT>torch.tensor,
            # we want to perform AST - torch.tensor
            other = other.wrap()
        elif isinstance(other, AdditiveSharingTensor) and isinstance(_self, torch.Tensor):
            # If we try to subtract a FPT>torch.tensor and a FPT>(wrap)>AST,
            # we swap operators so that we do the same operation as above
            _self, other = -other, -_self.wrap()

        response = getattr(_self, "sub")(other)
        response %= self.field  # Wrap around the field

        return response

    __sub__ = sub

    def sub_(self, value_or_tensor, tensor=None):
        if tensor is None:
            result = self.sub(value_or_tensor)
        else:
            result = self.sub(value_or_tensor * tensor)

        self.child = result.child
        return self

    def __isub__(self, other):
        self.child = self.sub(other).child

        return self

    @overloaded.method
    def t(self, _self, *args, **kwargs):
        """Transpose a tensor. Hooked is handled by the decorator"""
        response = getattr(_self, "t")(*args, **kwargs)

        return response

    def mul(self, other):
        """
        Hook manually mul to add the truncation part which is inherent to multiplication
        in the fixed precision setting
        """
        if isinstance(other, (int, torch.Tensor)):
            new_self = self.child
            new_other = other

        elif isinstance(self.child, (AdditiveSharingTensor, MultiPointerTensor)) and isinstance(
            other.child, torch.Tensor
        ):
            # If we try to multiply a FPT>AST with a FPT>torch.tensor,
            # we want to perform AST * torch.tensor
            new_self = self.child
            new_other = other

        elif isinstance(other.child, (AdditiveSharingTensor, MultiPointerTensor)) and isinstance(
            self.child, torch.Tensor
        ):
            # If we try to multiply a FPT>torch.tensor with a FPT>AST,
            # we swap operators so that we do the same operation as above
            new_self = other.child
            new_other = self

        elif isinstance(self.child, (AdditiveSharingTensor, MultiPointerTensor)) and isinstance(
            other.child, (AdditiveSharingTensor, MultiPointerTensor)
        ):
            # If we try to multiply a FPT>torch.tensor with a FPT>AST,
            # we swap operators so that we do the same operation as above
            new_self, new_other, _ = syft.frameworks.torch.hook_args.hook_method_args(
                "mul", self, other, None
            )

        elif isinstance(self.child, torch.Tensor) and isinstance(other.child, torch.Tensor):
            new_self, new_other, _ = syft.frameworks.torch.hook_args.hook_method_args(
                "mul", self, other, None
            )

            # To avoid problems when multiplying 2 negative numbers
            # we take absolute value of the operands

            # sgn_self is 1 when new_self is positive else it's 0
            sgn_self = (new_self < self.field // 2).long()
            pos_self = new_self * sgn_self
            neg_self = (self.field - new_self) * (1 - sgn_self)
            new_self = neg_self + pos_self

            # sgn_other is 1 when new_other is positive else it's 0
            sgn_other = (new_other < self.field // 2).long()
            pos_other = new_other * sgn_other
            neg_other = (self.field - new_other) * (1 - sgn_other)
            new_other = neg_other + pos_other

            # If both have the same sign, sgn is 1 else it's 0
            sgn = 1 - (sgn_self - sgn_other) ** 2

        # Send it to the appropriate class and get the response
        response = getattr(new_self, "mul")(new_other)

        # Put back SyftTensor on the tensors found in the response
        response = syft.frameworks.torch.hook_args.hook_response(
            "mul", response, wrap_type=type(self), wrap_args=self.get_class_attributes()
        )

        if not isinstance(other, (int, torch.Tensor)):
            response = response.truncate(self.precision_fractional, check_sign=False)
            response %= self.field  # Wrap around the field

            if isinstance(self.child, torch.Tensor) and isinstance(other.child, torch.Tensor):
                # Give back its sign to response
                pos_res = response * sgn
                neg_res = (self.field - response) * (1 - sgn)
                response = neg_res + pos_res

        else:
            response %= self.field  # Wrap around the field

        return response

    __mul__ = mul

    def __imul__(self, other):
        self = self.mul(other)
        return self

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

    def matmul(self, *args, **kwargs):
        """
        Hook manually matmul to add the truncation part which is inherent to multiplication
        in the fixed precision setting
        """

        other = args[0]

        if isinstance(other, FixedPrecisionTensor):
            assert (
                self.precision_fractional == other.precision_fractional
            ), "In matmul, all args should have the same precision_fractional"

        if isinstance(self.child, AdditiveSharingTensor) and isinstance(other.child, torch.Tensor):
            # If we try to matmul a FPT>(wrap)>AST with a FPT>torch.tensor,
            # we want to perform AST @ torch.tensor
            new_self = self.child
            new_args = (other,)
            new_kwargs = kwargs

        elif isinstance(other.child, AdditiveSharingTensor) and isinstance(
            self.child, torch.Tensor
        ):
            # If we try to matmul a FPT>torch.tensor with a FPT>(wrap)>AST,
            # we swap operators so that we do the same operation as above
            new_self = other.child
            new_args = (self,)
            new_kwargs = kwargs

        else:
            # Replace all syft tensor with their child attribute
            new_self, new_args, new_kwargs = syft.frameworks.torch.hook_args.hook_method_args(
                "matmul", self, args, kwargs
            )

        # Send it to the appropriate class and get the response
        response = getattr(new_self, "matmul")(*new_args, **new_kwargs)

        # Put back SyftTensor on the tensors found in the response
        response = syft.frameworks.torch.hook_args.hook_response(
            "matmul", response, wrap_type=type(self), wrap_args=self.get_class_attributes()
        )

        response %= self.field  # Wrap around the field
        response = response.truncate(other.precision_fractional)

        return response

    __matmul__ = matmul

    @overloaded.method
    def __gt__(self, _self, other):
        result = _self.__gt__(other)
        return result * self.base ** self.precision_fractional

    @overloaded.method
    def __ge__(self, _self, other):
        result = _self.__ge__(other)
        return result * self.base ** self.precision_fractional

    @overloaded.method
    def __lt__(self, _self, other):
        result = _self.__lt__(other)
        return result * self.base ** self.precision_fractional

    @overloaded.method
    def __le__(self, _self, other):
        result = _self.__le__(other)
        return result * self.base ** self.precision_fractional

    @overloaded.method
    def eq(self, _self, other):
        result = _self.eq(other)
        return result * self.base ** self.precision_fractional

    @staticmethod
    @overloaded.module
    def torch(module):
        def add(self, other):
            return self.__add__(other)

        module.add = add

        def sub(self, other):
            return self.__sub__(other)

        module.sub = sub

        def mul(self, other):
            return self.__mul__(other)

        module.mul = mul

        def matmul(self, other):
            return self.matmul(other)

        module.matmul = matmul

        def addmm(bias, input_tensor, weight):
            matmul = input_tensor.matmul(weight)
            result = bias.add(matmul)
            return result

        module.addmm = addmm

        def dot(self, other):
            return self.__mul__(other).sum()

        module.dot = dot

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

            # Change to tuple if not one
            stride = torch.nn.modules.utils._pair(stride)
            padding = torch.nn.modules.utils._pair(padding)
            dilation = torch.nn.modules.utils._pair(dilation)

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
            # It basically is the position of the values used for the top left convolution.
            pattern_ind = []
            for ch in range(nb_channels_in):
                for r in range(nb_rows_kernel):
                    for c in range(nb_cols_kernel):
                        pixel = r * nb_cols_in * dilation[0] + c * dilation[1]
                        pattern_ind.append(pixel + ch * nb_rows_in * nb_cols_in)

            # The image tensor is reshaped for the matrix multiplication:
            # on each row of the new tensor will be the input values used for each filter convolution
            # We will get a matrix [[in values to compute out value 0],
            #                       [in values to compute out value 1],
            #                       ...
            #                       [in values to compute out value nb_rows_out*nb_cols_out]]
            im_flat = input.view(batch_size, -1)
            im_reshaped = []
            for cur_row_out in range(nb_rows_out):
                for cur_col_out in range(nb_cols_out):
                    # For each new output value, we just need to shift the receptive field
                    offset = cur_row_out * stride[0] * nb_cols_in + cur_col_out * stride[1]
                    tmp = [ind + offset for ind in pattern_ind]
                    im_reshaped.append(im_flat[:, tmp].wrap())
            im_reshaped = torch.stack(im_reshaped).permute(1, 0, 2)

            # The convolution kernels are also reshaped for the matrix multiplication
            # We will get a matrix [[weights for out channel 0],
            #                       [weights for out channel 1],
            #                       ...
            #                       [weights for out channel nb_channels_out]].TRANSPOSE()
            weight_reshaped = weight.view(nb_channels_out // groups, -1).t().wrap()

            # Now that everything is set up, we can compute the result
            if groups > 1:
                res = []
                chunks_im = torch.chunk(im_reshaped, groups, dim=2)
                chunks_weights = torch.chunk(weight_reshaped, groups, dim=0)
                for g in range(groups):
                    tmp = chunks_im[g].matmul(chunks_weights[g])
                    res.append(tmp)
                res = torch.cat(res, dim=2)
            else:
                res = im_reshaped.matmul(weight_reshaped)

            # Add a bias if needed
            if bias is not None:
                res += bias

            # ... And reshape it back to an image
            res = (
                res.permute(0, 2, 1)
                .view(batch_size, nb_channels_out, nb_rows_out, nb_cols_out)
                .contiguous()
            )
            return res.child

        module.conv2d = conv2d

        # You can also overload functions in submodules!
        @overloaded.module
        def nn(module):
            """
            The syntax is the same, so @overloaded.module handles recursion
            Note that we don't need to add the @staticmethod decorator
            """

            @overloaded.module
            def functional(module):
                def linear(*args):
                    """
                    Un-hook the function to have its detailed behaviour
                    """
                    return torch.nn.functional.native_linear(*args)

                module.linear = linear

            module.functional = functional

        # Modules should be registered just like functions
        module.nn = nn

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

        tensor = args[0] if not isinstance(args[0], tuple) else args[0][0]

        # Check that the function has not been overwritten
        try:
            # Try to get recursively the attributes in cmd = "<attr1>.<attr2>.<attr3>..."
            cmd = cls.rgetattr(cls, cmd)
            return cmd(*args, **kwargs)
        except AttributeError:
            pass

        # TODO: I can't manage the import issue, can you?
        # Replace all FixedPrecisionTensor with their child attribute
        new_args, new_kwargs, new_type = syft.frameworks.torch.hook_args.hook_function_args(
            cmd, args, kwargs
        )

        # build the new command
        new_command = (cmd, None, new_args, new_kwargs)

        # Send it to the appropriate class and get the response
        response = new_type.handle_func_command(new_command)

        # Put back FixedPrecisionTensor on the tensors found in the response
        response = syft.frameworks.torch.hook_args.hook_response(
            cmd, response, wrap_type=cls, wrap_args=tensor.get_class_attributes()
        )

        return response

    def share(self, *owners, field=None, crypto_provider=None):
        if field is None:
            field = self.field
        else:
            assert (
                field == self.field
            ), "When sharing a FixedPrecisionTensor, the field of the resulting AdditiveSharingTensor \
                must be the same as the one of the original tensor"
        self.child = self.child.share(
            *owners, field=field, crypto_provider=crypto_provider, no_wrap=True
        )
        return self

    @staticmethod
    def simplify(tensor: "FixedPrecisionTensor") -> tuple:
        """Takes the attributes of a FixedPrecisionTensor and saves them in a tuple.

        Args:
            tensor: a FixedPrecisionTensor.

        Returns:
            tuple: a tuple holding the unique attributes of the fixed precision tensor.
        """
        chain = None
        if hasattr(tensor, "child"):
            chain = syft.serde._simplify(tensor.child)

        return (
            syft.serde._simplify(tensor.id),
            tensor.field,
            tensor.base,
            tensor.precision_fractional,
            tensor.kappa,
            syft.serde._simplify(tensor.tags),
            syft.serde._simplify(tensor.description),
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

        tensor_id, field, base, precision_fractional, kappa, tags, description, chain = tensor_tuple

        tensor = FixedPrecisionTensor(
            owner=worker,
            id=syft.serde._detail(worker, tensor_id),
            field=field,
            base=base,
            precision_fractional=precision_fractional,
            kappa=kappa,
            tags=syft.serde._detail(worker, tags),
            description=syft.serde._detail(worker, description),
        )

        if chain is not None:
            chain = syft.serde._detail(worker, chain)
            tensor.child = chain

        return tensor
