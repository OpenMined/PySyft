import syft
import torch
from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
from syft.frameworks.torch.overload_torch import overloaded


class FixedPrecisionTensor(AbstractTensor):
    def __init__(
        self,
        owner=None,
        id=None,
        field: int = (2 ** 62) - 1,
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
        self.torch_max_value = torch.tensor([round(self.field / 2)])

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

    def fix_precision(self):
        """This method encodes the .child object using fixed precision"""

        rational = self.child

        upscaled = (rational * self.base ** self.precision_fractional).long()
        field_element = upscaled % self.field
        field_element.owner = rational.owner

        # Handle neg values
        gate = field_element.gt(self.torch_max_value).long()
        neg_nums = (field_element - self.field) * gate
        pos_nums = field_element * (1 - gate)
        field_element = neg_nums + pos_nums

        self.child = field_element
        return self

    def float_precision(self):
        """this method returns a new tensor which has the same values as this
        one, encoded with floating point precision"""

        value = self.child.long() % self.field

        gate = value.native_gt(self.torch_max_value).long()
        neg_nums = (value - self.field) * gate
        pos_nums = value * (1 - gate)
        result = (neg_nums + pos_nums).float() / (self.base ** self.precision_fractional)

        return result

    def truncate(self, precision_fractional):
        truncation = self.base ** precision_fractional
        self.child /= truncation
        return self

    @overloaded.method
    def add(self, _self, *args, **kwargs):
        """Add two fixed precision tensors together.
        """
        response = getattr(_self, "add")(*args, **kwargs)

        return response

    __add__ = add

    def __iadd__(self, other):
        """Add two fixed precision tensors together.
        """
        self.child = self.add(other).child

        return self

    @overloaded.method
    def t(self, _self, *args, **kwargs):
        """Transpose a tensor. Hooked is handled by the decorator"""
        response = getattr(_self, "t")(*args, **kwargs)

        return response

    def mul(self, *args, **kwargs):
        """
        Hook manually mul to add the truncation part which is inherent to multiplication
        in the fixed precision setting
        """
        # Replace all syft tensor with their child attribute
        new_self, new_args, new_kwargs = syft.frameworks.torch.hook_args.hook_method_args(
            "mul", self, args, kwargs
        )

        # Send it to the appropriate class and get the response
        response = getattr(new_self, "mul")(*new_args, **new_kwargs)

        # Put back SyftTensor on the tensors found in the response
        response = syft.frameworks.torch.hook_args.hook_response(
            "mul", response, wrap_type=type(self), wrap_args=self.get_class_attributes()
        )

        other = args[0]

        assert (
            self.precision_fractional == other.precision_fractional
        ), "In mul, all args should have the same precision_fractional"

        return response.truncate(other.precision_fractional)

    __mul__ = mul

    def matmul(self, *args, **kwargs):
        """
        Hook manually matmul to add the truncation part which is inherent to multiplication
        in the fixed precision setting
        """
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

        other = args[0]

        assert (
            self.precision_fractional == other.precision_fractional
        ), "In matmul, all args should have the same precision_fractional"

        return response.truncate(other.precision_fractional)

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
        def mul(self, other):
            return self.__mul__(other)

        module.mul = mul

        def addmm(bias, input_tensor, weight):
            matmul = input_tensor.matmul(weight)
            result = bias.add(matmul)
            return result

        module.addmm = addmm

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

    def get(self):
        """Just a pass through. This is most commonly used when calling .get() on a
        FixedPrecisionTensor which has also been shared."""
        class_attributes = self.get_class_attributes()
        return FixedPrecisionTensor(
            **class_attributes,
            owner=self.owner,
            tags=self.tags,
            description=self.description,
            id=self.id,
        ).on(self.child.get())

    def share(self, *owners, field=None, crypto_provider=None):
        self.child = self.child.share(*owners, field=field, crypto_provider=crypto_provider)
        return self
