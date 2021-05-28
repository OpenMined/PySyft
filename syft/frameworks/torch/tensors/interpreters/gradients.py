from .gradients_core import GradFunc
from .gradients_core import apply_dim_transformations

import torch

import syft as sy


class AddBackward(GradFunc):
    def __init__(self, self_, other):
        super().__init__(self, self_, other)
        self.self_ = self_
        self.other = other

    def gradient(self, grad):
        grad_self = grad.copy()
        grad_other = grad.copy() if isinstance(self.self_, type(self.other)) else None

        if not isinstance(self.other.child, int):
            if self.self_.shape != self.other.shape:
                grad_self, grad_other = apply_dim_transformations(
                    grad_self, grad_other, self.self_.shape, self.other.shape
                )

        return (grad_self, grad_other)


class SubBackward(GradFunc):
    def __init__(self, self_, other):
        super().__init__(self, self_, other)
        self.self_ = self_
        self.other = other

    def gradient(self, grad):
        grad_self = grad.copy()
        grad_other = grad * -1 if isinstance(self.self_, type(self.other)) else None

        if not isinstance(self.other.child, int):
            if self.self_.shape != self.other.shape:
                grad_self, grad_other = apply_dim_transformations(
                    grad_self, grad_other, self.self_.shape, self.other.shape
                )
        return (grad_self, grad_other)


class SumBackward(GradFunc):
    """Tensor Sum backward gradient class"""

    def __init__(self, *args, **kwargs):
        super().__init__(self, args[0])

        # preprocess inputs:
        assert len(args) >= 1
        if len(args) == 1:
            (input,) = args  # no dimension to sum over in args
            dim = kwargs.get("dim", None)
        else:
            assert len(args) == 2
            assert "dim" not in kwargs
            input, dim = args  # dimension to sum over in args

        keepdim = kwargs.get("keepdim", False)

        self.self_ = input
        self.dim = dim
        self.keepdim = keepdim

    def gradient(self, grad):
        if self.keepdim:
            raise NotImplementedError("keepdim arg in sum() is not supported in autograd currently")

        if not self.keepdim and self.dim is not None:
            grad = grad.unsqueeze(self.dim)

        if grad.shape != self.self_.shape:
            grad = grad.reshape([-1, 1])
        r = ((self.self_ * 0 + 1) * grad,)
        return r


class MeanBackward(GradFunc):
    """Tensor Mean backward gradient class"""

    def __init__(self, self_, dim=None):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        if grad.shape != self.self_.shape:
            grad = grad.reshape([-1, 1])
        numel = self.self_.numel()
        return ((self.self_ * 0 + 1) * grad / numel,)


class ReshapeBackward(GradFunc):
    """Tensor reshape backward gradient class"""

    def __init__(self, self_, *dims):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        if grad.shape != self.self_.shape:
            grad = grad.reshape(self.self_.shape)
        return ((self.self_ * 0 + 1) * grad,)


class ViewBackward(GradFunc):
    """Tensor reshape backward gradient class"""

    def __init__(self, self_, *dims):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        if grad.shape != self.self_.shape:
            grad = grad.view(self.self_.shape)
        return ((self.self_ * 0 + 1) * grad,)


class AsinBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        grad_self_ = grad * (-self.self_ * self.self_ + 1).rsqrt()
        return (grad_self_,)


class LogBackward(GradFunc):
    """Log backward gradient class"""

    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        grad_self_ = grad * (1 / self.self_)
        return (grad_self_,)


class ExpBackward(GradFunc):
    """Exp backward gradient class"""

    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        grad_self_ = grad * self.self_.exp()
        return (grad_self_,)


class MulBackward(GradFunc):
    def __init__(self, self_, other):
        super().__init__(self, self_, other)
        self.self_ = self_
        self.other = other

    def gradient(self, grad):
        grad_self_ = grad * self.other
        grad_other = grad * self.self_ if isinstance(self.self_, type(self.other)) else None
        return (grad_self_, grad_other)


class NegBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        grad_self_ = grad * -1
        return (grad_self_,)


class DivBackward(GradFunc):
    def __init__(self, self_, other):
        super().__init__(self, self_, other)
        self.self_ = self_
        self.other = other

    def gradient(self, grad):
        # assert isinstance(self.other, int)
        grad_self_ = grad / self.other
        return (grad_self_,)


class PowBackward(GradFunc):
    def __init__(self, self_, power):
        super().__init__(self, self_, power)
        self.self_ = self_
        self.power = power

    def gradient(self, grad):
        power = self.power
        return (power * self.self_ ** (power - 1) * grad,)


class MatmulBackward(GradFunc):
    def __init__(self, self_, other):
        super().__init__(self, self_, other)
        self.self_ = self_
        self.other = other

    def gradient(self, grad):
        grad_self_ = grad @ self.other.t()
        grad_other = self.self_.t() @ grad if isinstance(self.self_, type(self.other)) else None
        return (grad_self_, grad_other)


class TBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        return (grad.t(),)


class SigmoidBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        grad_self_ = grad * self.self_.sigmoid() * (1 - self.self_.sigmoid())
        return (grad_self_,)


class SinBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        grad_self_ = grad * self.self_.cos()
        return (grad_self_,)


class SinhBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        grad_self_ = grad * self.self_.cosh()
        return (grad_self_,)


# class SqrtBackward(GradFunc):
#     def __init__(self, self_):
#         super().__init__(self, self_)
#         self.self_ = self_
#
#     def gradient(self, grad):
#         TODO: Broken as of Garbage Collection for `AutoGradTensor` (#3387)
#         grad_self_ = grad / (2 * self.result)
#         return (grad_self_,)


class TanhBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        grad_self_ = grad * (1 - self.self_.tanh() ** 2)
        return (grad_self_,)


class ReluBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        zero = self.self_ * 0
        gt_zero = self.self_ > zero
        result = gt_zero * grad
        return (result,)


# class Log_softmaxBackward(GradFunc):
#     """Log Softmax backward gradient class"""
#
#     def __init__(self, output, target):
#         super().__init__(self, output, target)
#         self.output = output
#         self.other = target
#
#     def forward(self, output, target):
#         assert False
#         maximum_value = output.max()
#         x = output - maximum_value
#         exp_x = torch.exp(x)
#         sum_x = torch.sum(exp_x, dim=1)
#         x_klass = (x * target).sum(dim=1)
#         probs = x_klass - torch.log(sum_x)
#         self.probs = probs
#         return probs
#
#     def gradient(self, grad):
#         raise NotImplementedError
#
#
# class SoftmaxBackward(GradFunc):
#     """Softmax backward gradient class"""
#
#     def __init__(self, input, dim):
#         super().__init__(self, input)
#
#     def forward(self, input, dim):
#         maximum_value = input.max(dim, keepdim=True)[0]
#         logits = input - maximum_value
#         numerator = logits.exp()
#         inv_denominator = numerator.sum(dim, keepdim=True).reciprocal()
#         probs = numerator * inv_denominator
#         self.probs = probs
#         return probs
#
#     def gradient(self, grad):
#         probs = self.probs
#         dim = self.dim
#         result = grad.add(-probs.mul(grad).sum(dim, keepdim=True)).mul(probs)
#         return (result,)


class Cross_entropyBackward(GradFunc):
    """Cross Entropy backward gradient class"""

    def __init__(self, output, target):
        super().__init__(self, output, target)
        self.output = output
        self.other = target

    def forward(self, pred, target):
        def softmax(input, dim):
            ast_mode = isinstance(input.child, sy.AdditiveSharingTensor)
            fp_mode = not ast_mode and isinstance(input, sy.FixedPrecisionTensor)

            fp_kwargs = input.get_class_attributes()
            ast_kwargs = input.child.get_class_attributes()
            ast_locations = input.child.locations

            def decode(x):
                if ast_mode:
                    return x.copy().get().float_prec()
                elif fp_mode:
                    return x.copy().float_precision()
                else:
                    return x.copy()

            def encode(x):
                return x.fix_precision(**fp_kwargs).share(*ast_locations, **ast_kwargs).child

            # i = decode(input)
            # print("\n#0  input softmax", i.shape, i.abs().mean(), i.abs().max())

            zeros = [torch.tensor([1.0])]

            while (zeros[0] != 0).any():

                if ast_mode:
                    maximum_value = input.max(dim, keepdim=True)
                else:
                    maximum_value = input.max(dim, keepdim=True)[0]

                # a = decode(maximum_value)
                # print("#$ maximum", a.shape, a.mean(), a.min(), a.max())

                logits = input - maximum_value
                # l = decode(logits)
                # print("#1 logits", l.shape, l.mean(), l.min(), l.max())

                zeros[0] = decode(logits.max(dim, keepdim=True))

                if (zeros[0] != 0).any():
                    # print(
                    #     "********************************************* RETRY LOGITS ***********************************************"  # noqa
                    # )
                    n = input.shape[dim]
                    scaler = (100 + torch.tensor(range(n)).float().reshape(1, -1) / n) / 100
                    input *= encode(scaler)

            numerator = logits.exp()

            # b = decode(numerator)
            # print("#2 numerator", b.shape, b.mean(), b.min(), b.max())

            numerator_sum = numerator.sum(dim, keepdim=True)

            # bs = decode(numerator_sum)
            # print("#* numerator sum", bs.shape, bs.mean(), bs.min(), bs.max())

            inv_denominator = numerator_sum.reciprocal()
            # c = decode(inv_denominator)
            # print("#3 inv_den", c.shape, c.mean(), c.min(), c.max())

            probs = numerator * inv_denominator
            return probs

        softmax = softmax(pred, 1)
        loss_values = -softmax.log(input_in_01=True).mul(target)

        self.softmax = softmax
        self.target = target

        return loss_values.sum() / target.shape[0]

    def gradient(self, grad):
        softmax = sy.AutogradTensor()
        softmax.child = self.softmax
        target = sy.AutogradTensor()
        target.child = self.target

        loss_grad = softmax.sub(target)
        batch_size = target.shape[0]
        result = loss_grad / batch_size * grad
        return (result,)


class Max_pool2dBackward(GradFunc):
    def __init__(
        self,
        input,
        kernel_size,
        padding=0,
        stride=2,
        dilation=1,
        ceil_mode=False,
        return_indices=False,
    ):
        super().__init__(
            self, input, kernel_size, padding, stride, dilation, ceil_mode, return_indices
        )
        self.input_shape = input.shape
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(
        self,
        input,
        kernel_size,
        padding=0,
        stride=2,
        dilation=1,
        ceil_mode=False,
        return_indices=False,
    ):
        if return_indices:
            raise NotImplementedError(
                "We don't currently support return more than one tensor from autograded methods"
            )

        result, indices = input.max_pool2d(
            kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=True,
        )
        self.indices = indices
        return result

    def gradient(self, grad_output):
        output_size = self.input_shape
        indices = self.indices
        kernel_size = self.kernel_size
        padding = self.padding
        stride = self.stride
        dilation = self.dilation
        ceil_mode = self.ceil_mode
        # convert to tuple
        padding = torch.nn.modules.utils._pair(padding)
        stride = torch.nn.modules.utils._pair(stride)
        dilation = torch.nn.modules.utils._pair(dilation)
        assert padding[0] == padding[1], "padding must be same in all axes"
        grad = grad_output._max_pool2d_backward(
            indices,
            kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            ceil_mode=ceil_mode,
            output_size=output_size,
        )
        return (grad,)


class Conv2dBackward(GradFunc):
    def __init__(self, input, weight, bias, stride, padding, dilation, groups):
        super().__init__(self, input, weight, bias, stride, padding, dilation, groups)
        padding = torch.nn.modules.utils._pair(padding)
        stride = torch.nn.modules.utils._pair(stride)
        dilation = torch.nn.modules.utils._pair(dilation)
        self.input = input
        self.kernel = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def gradient(self, grad_output):
        def _grad_input_padding(
            grad_output, input_size, stride, padding, kernel_size, dilation=None
        ):
            if dilation is None:
                # For backward compatibility
                print("_grad_input_padding 'dilation' argument not provided. Default of 1 is used.")
                dilation = [1] * len(stride)

            input_size = list(input_size)
            k = grad_output.dim() - 2

            if len(input_size) == k + 2:
                input_size = input_size[-k:]
            if len(input_size) != k:
                raise ValueError(
                    "input_size must have {} elements (got {})".format(k + 2, len(input_size))
                )

            def dim_size(d):
                return (
                    (grad_output.shape[d + 2] - 1) * stride[d]
                    - 2 * padding[d]
                    + 1
                    + dilation[d] * (kernel_size[d] - 1)
                )

            min_sizes = [dim_size(d) for d in range(k)]
            max_sizes = [min_sizes[d] + stride[d] - 1 for d in range(k)]
            for size, min_size, max_size in zip(input_size, min_sizes, max_sizes):
                if size < min_size or size > max_size:
                    raise ValueError(
                        (
                            "requested an input grad size of {}, but valid sizes range "
                            "from {} to {} (for a grad_output of {})"
                        ).format(input_size, min_sizes, max_sizes, grad_output.shape[2:])
                    )

            return tuple(input_size[d] - min_sizes[d] for d in range(k))

        # get input, kernel, and sizes:
        input, kernel, padding, stride, dilation, groups = (
            self.input,
            self.kernel,
            self.padding,
            self.stride,
            self.dilation,
            self.groups,
        )
        batch_size = input.shape[0]
        out_channels, in_channels, kernel_size_y, kernel_size_x = kernel.shape
        in_channels *= groups
        assert input.shape[1] == in_channels, "wrong number of input channels"
        assert grad_output.shape[1] == out_channels, "wrong number of output channels"
        assert grad_output.shape[0] == batch_size, "wrong batch size"

        # TODO: Implement conv2d gradient under following condition:
        if groups > 1 and input.shape[1] > groups:
            raise NotImplementedError(
                "conv2d backward with groups > 1 and in_channels > groups not implemented"
            )

        # compute gradient with respect to input:
        # TODO: Eliminate dependency on torch internal function by implementing in util
        output_padding = _grad_input_padding(
            grad_output,
            input.shape,
            stride,
            padding,
            (kernel_size_y, kernel_size_x),
            dilation=dilation,
        )
        grad_input = grad_output.conv_transpose2d(
            kernel,
            bias=None,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        )

        # compute gradient with respect to kernel:
        grad_output = grad_output.repeat(1, in_channels // groups, 1, 1)
        grad_output = grad_output.view(
            grad_output.shape[0] * grad_output.shape[1],
            1,
            grad_output.shape[2],
            grad_output.shape[3],
        )
        input = input.view(1, input.shape[0] * input.shape[1], input.shape[2], input.shape[3])
        # dilation and stride are swapped based on PyTorch's conv2d_weight implementation
        grad_kernel = input.conv2d(
            grad_output,
            bias=None,
            stride=dilation,
            padding=padding,
            dilation=stride,
            groups=in_channels * batch_size,
        )
        grad_kernel = grad_kernel.view(
            batch_size,
            grad_kernel.shape[1] // batch_size,
            grad_kernel.shape[2],
            grad_kernel.shape[3],
        )
        grad_kernel = (
            grad_kernel.sum(0)
            .view(
                in_channels // groups,
                out_channels,
                grad_kernel.shape[2],
                grad_kernel.shape[3],
            )
            .transpose(0, 1)
        )
        grad_kernel = grad_kernel.narrow(2, 0, kernel_size_y)
        grad_kernel = grad_kernel.narrow(3, 0, kernel_size_x)
        return (grad_input, grad_kernel)
