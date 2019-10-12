import torch

import syft
from syft.workers.abstract import AbstractWorker
from syft.generic.frameworks.hook import hook_args
from syft.generic.pointers.multi_pointer import MultiPointerTensor
from syft.generic.tensor import AbstractTensor
from syft.frameworks.torch.tensors.interpreters.additive_shared import AdditiveSharingTensor
from syft.generic.frameworks.overload import overloaded


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
        self.id = id if id else syft.ID_PROVIDER.pop()
        self.child = None

        self.field = field
        self.base = base
        self.precision_fractional = precision_fractional
        self.kappa = kappa

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

    def backward(self, *args, **kwargs):
        """Calling backward on Precision Tensor doesn't make sense, but sometimes a call
        can be propagated downward the chain to an Precision Tensor (for example in
        create_grad_objects), so we just ignore the call."""
        pass

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
        torch_max_value = torch.tensor(self.field).long()

        gate = value.native_gt(torch_max_value / 2).long()
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
            torch_max_value = torch.tensor(self.field).long()
            gate = self.child.native_gt(torch_max_value / 2).long()
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

    def mul_and_div(self, other, cmd):
        """
        Hook manually mul and div to add the trucation/rescaling part
        which is inherent to these operations in the fixed precision setting
        """
        changed_sign = False
        if isinstance(other, FixedPrecisionTensor):
            assert (
                self.precision_fractional == other.precision_fractional
            ), "In mul and div, all args should have the same precision_fractional"
            assert self.base == other.base, "In mul and div, all args should have the same base"

        if isinstance(other, (int, torch.Tensor, AdditiveSharingTensor)):
            new_self = self.child
            new_other = other

        elif isinstance(self.child, (AdditiveSharingTensor, MultiPointerTensor)) and isinstance(
            other.child, torch.Tensor
        ):
            # If operands are FPT>AST and FPT>torch.tensor,
            # we want to perform the operation on AST and torch.tensor
            if cmd == "mul":
                new_self = self.child
            elif cmd == "div":
                new_self = self.child * self.base ** self.precision_fractional
            new_other = other

        elif isinstance(other.child, (AdditiveSharingTensor, MultiPointerTensor)) and isinstance(
            self.child, torch.Tensor
        ):
            # If operands are FPT>torch.tensor and FPT>AST,
            # we swap operators so that we do the same operation as above
            if cmd == "mul":
                new_self = other.child
                new_other = self
            elif cmd == "div":
                # TODO how to divide by AST?
                raise NotImplementedError(
                    "Division of a FixedPrecisionTensor by an AdditiveSharingTensor not implemented"
                )

        elif (
            cmd == "mul"
            and isinstance(self.child, (AdditiveSharingTensor, MultiPointerTensor))
            and isinstance(other.child, (AdditiveSharingTensor, MultiPointerTensor))
        ):
            # If we try to multiply a FPT>torch.tensor with a FPT>AST,
            # we swap operators so that we do the same operation as above
            new_self, new_other, _ = hook_args.unwrap_args_from_method("mul", self, other, None)

        else:
            # Replace all syft tensor with their child attribute
            new_self, new_other, _ = hook_args.unwrap_args_from_method(cmd, self, other, None)

            # To avoid problems with negative numbers
            # we take absolute value of the operands
            # The problems could be 1) bad truncation for multiplication
            # 2) overflow when scaling self in division

            # sgn_self is 1 when new_self is positive else it's 0
            # The comparison is different is new_self is a torch tensor or an AST
            sgn_self = (
                (new_self < self.field // 2).long()
                if isinstance(new_self, torch.Tensor)
                else new_self > 0
            )
            pos_self = new_self * sgn_self
            neg_self = (
                (self.field - new_self) * (1 - sgn_self)
                if isinstance(new_self, torch.Tensor)
                else new_self * (sgn_self - 1)
            )
            new_self = neg_self + pos_self

            # sgn_other is 1 when new_other is positive else it's 0
            # The comparison is different is new_other is a torch tensor or an AST
            sgn_other = (
                (new_other < self.field // 2).long()
                if isinstance(new_other, torch.Tensor)
                else new_other > 0
            )
            pos_other = new_other * sgn_other
            neg_other = (
                (self.field - new_other) * (1 - sgn_other)
                if isinstance(new_other, torch.Tensor)
                else new_other * (sgn_other - 1)
            )
            new_other = neg_other + pos_other

            # If both have the same sign, sgn is 1 else it's 0
            # To be able to write sgn = 1 - (sgn_self - sgn_other) ** 2,
            # we would need to overload the __add__ for operators int and AST.
            sgn = -(sgn_self - sgn_other) ** 2 + 1
            changed_sign = True

            if cmd == "div":
                new_self *= self.base ** self.precision_fractional

        # Send it to the appropriate class and get the response
        response = getattr(new_self, cmd)(new_other)

        # Put back SyftTensor on the tensors found in the response
        response = hook_args.hook_response(
            cmd, response, wrap_type=type(self), wrap_args=self.get_class_attributes()
        )

        if not isinstance(other, (int, torch.Tensor, AdditiveSharingTensor)):
            if cmd == "mul":
                # If operation is mul, we need to truncate
                response = response.truncate(self.precision_fractional, check_sign=False)

            response %= self.field  # Wrap around the field

            if changed_sign:
                # Give back its sign to response
                pos_res = response * sgn
                neg_res = response * (sgn - 1)
                response = neg_res + pos_res

        else:
            response %= self.field  # Wrap around the field

        return response

    def mul(self, other):
        return self.mul_and_div(other, "mul")

    __mul__ = mul

    def __imul__(self, other):
        self = self.mul_and_div(other, "mul")
        return self

    mul_ = __imul__

    def div(self, other):
        return self.mul_and_div(other, "div")

    __truediv__ = div

    def __itruediv__(self, other):
        self = self.mul_and_div(other, "div")
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
            new_self, new_args, new_kwargs = hook_args.unwrap_args_from_method(
                "matmul", self, args, kwargs
            )

        # Send it to the appropriate class and get the response
        response = getattr(new_self, "matmul")(*new_args, **new_kwargs)

        # Put back SyftTensor on the tensors found in the response
        response = hook_args.hook_response(
            "matmul", response, wrap_type=type(self), wrap_args=self.get_class_attributes()
        )

        response %= self.field  # Wrap around the field
        response = response.truncate(other.precision_fractional)

        return response

    __matmul__ = matmul
    mm = matmul

    @overloaded.method
    def __gt__(self, _self, other):
        result = _self.__gt__(other)
        return result.long() * self.base ** self.precision_fractional

    @overloaded.method
    def __ge__(self, _self, other):
        result = _self.__ge__(other)
        return result.long() * self.base ** self.precision_fractional

    @overloaded.method
    def __lt__(self, _self, other):
        result = _self.__lt__(other)
        return result.long() * self.base ** self.precision_fractional

    @overloaded.method
    def __le__(self, _self, other):
        result = _self.__le__(other)
        return result.long() * self.base ** self.precision_fractional

    @overloaded.method
    def eq(self, _self, other):
        result = _self.eq(other)
        return result.long() * self.base ** self.precision_fractional

    __eq__ = eq

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

        def div(self, other):
            return self.__truediv__(other)

        module.div = div

        def matmul(self, other):
            return self.matmul(other)

        module.matmul = matmul
        module.mm = matmul

        def addmm(bias, input_tensor, weight):
            matmul = input_tensor.matmul(weight)
            result = bias.add(matmul)
            return result

        module.addmm = addmm

        def sigmoid(tensor):
            """
            Overloads torch.sigmoid to be able to use MPC
            Approximation with polynomial interpolation of degree 5 over [-8,8]
            Ref: https://mortendahl.github.io/2017/04/17/private-deep-learning-with-mpc/#approximating-sigmoid
            """

            weights = [0.5, 1.91204779e-01, -4.58667307e-03, 4.20690803e-05]
            degrees = [0, 1, 3, 5]

            max_degree = degrees[-1]
            max_idx = degrees.index(max_degree)

            # initiate with term of degree 0 to avoid errors with tensor ** 0
            result = (tensor * 0 + 1) * torch.tensor(weights[0]).fix_precision().child
            for w, d in zip(weights[1:max_idx], degrees[1:max_idx]):
                result += (tensor ** d) * torch.tensor(w).fix_precision().child

            return result

        module.sigmoid = sigmoid

        def tanh(tensor):
            """
            Overloads torch.tanh to be able to use MPC
            """

            result = 2 * sigmoid(2 * tensor) - 1

            return result

        module.tanh = tanh

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
            The idea is to build new tensors from input and weight to compute a
            matrix multiplication equivalent to the convolution.

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
            # Currently, kwargs are not unwrapped by hook_args
            # So this needs to be done manually
            if bias.is_wrapper:
                bias = bias.child

            assert len(input.shape) == 4
            assert len(weight.shape) == 4

            # Change to tuple if not one
            stride = torch.nn.modules.utils._pair(stride)
            padding = torch.nn.modules.utils._pair(padding)
            dilation = torch.nn.modules.utils._pair(dilation)

            # Extract a few useful values
            batch_size, nb_channels_in, nb_rows_in, nb_cols_in = input.shape
            nb_channels_out, nb_channels_kernel, nb_rows_kernel, nb_cols_kernel = weight.shape

            if bias is not None:
                assert len(bias) == nb_channels_out

            # Check if inputs are coherent
            assert nb_channels_in == nb_channels_kernel * groups
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
                    im_reshaped.append(im_flat[:, tmp])
            im_reshaped = torch.stack(im_reshaped).permute(1, 0, 2)

            # The convolution kernels are also reshaped for the matrix multiplication
            # We will get a matrix [[weights for out channel 0],
            #                       [weights for out channel 1],
            #                       ...
            #                       [weights for out channel nb_channels_out]].TRANSPOSE()
            weight_reshaped = weight.view(nb_channels_out // groups, -1).t()

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

        tensor = args[0] if not isinstance(args[0], (tuple, list)) else args[0][0]

        # Check that the function has not been overwritten
        try:
            # Try to get recursively the attributes in cmd = "<attr1>.<attr2>.<attr3>..."
            cmd = cls.rgetattr(cls, cmd)
            return cmd(*args, **kwargs)
        except AttributeError:
            pass

        # Replace all FixedPrecisionTensor with their child attribute
        new_args, new_kwargs, new_type = hook_args.unwrap_args_from_function(cmd, args, kwargs)

        # build the new command
        new_command = (cmd, None, new_args, new_kwargs)

        # Send it to the appropriate class and get the response
        response = new_type.handle_func_command(new_command)

        # Put back FixedPrecisionTensor on the tensors found in the response
        response = hook_args.hook_response(
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


### Register the tensor with hook_args.py ###
hook_args.default_register_tensor(FixedPrecisionTensor)
