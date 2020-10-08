import torch

import syft as sy

from syft.generic.frameworks.types import FrameworkTensor
from syft.generic.utils import allow_command
from syft.generic.utils import remote


def linear(*args):
    """
    Un-hook the function to have its detailed behaviour
    """
    return torch.nn.functional.native_linear(*args)


def dropout(input, p=0.5, training=True, inplace=False):
    """
    Args:
        p: probability of an element to be zeroed. Default: 0.5
        training: If training, cause dropout layers are not used during evaluation of model
        inplace: If set to True, will do this operation in-place. Default: False
    """

    if training:
        binomial = torch.distributions.binomial.Binomial(probs=1 - p)

        # we must convert the normal tensor to fixed precision before multiplication
        # Note that: Weights of a model are alwasy Float values
        # Hence input will always be of type FixedPrecisionTensor > ...
        noise = (binomial.sample(input.shape).type(torch.FloatTensor) * (1.0 / (1.0 - p))).fix_prec(
            **input.get_class_attributes(), no_wrap=True
        )

        if inplace:
            input = input * noise
            return input

        return input * noise

    return input


def batch_norm(
    input, running_mean, running_var, weight, bias, training, exponential_average_factor, eps
):
    """
    Implementation of batch_norm

    Note that exponential_average_factor is not supported for the moment and should be set to 0
    when training a model. However for testing, this doesn't matter.
    """
    input = input.permute(1, 0, 2, 3)
    input_shape = input.shape
    input = input.reshape(input_shape[0], -1)
    input = input.t()

    if training:
        assert exponential_average_factor == 0  # == momentum
        mean = input.mean(dim=0)
        var = input.var(dim=0) + eps
    else:
        mean = running_mean
        var = running_var

    x = var.reciprocal(method="newton")

    normalized = x * (input - mean)
    result = normalized * weight + bias

    result = result.t()
    result = result.reshape(*input_shape)
    result = result.permute(1, 0, 2, 3)

    return result


@allow_command
def _pre_conv(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    This is a block of local computation done at the beginning of the convolution. It
    basically does the matrix unrolling to be able to do the convolution as a simple
    matrix multiplication.

    Because all the computation are local, we add the @allow_command and run it directly
    on each share of the additive sharing tensor, when running mpc computations
    """
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
        ((nb_rows_in + 2 * padding[0] - dilation[0] * (nb_rows_kernel - 1) - 1) / stride[0]) + 1
    )
    nb_cols_out = int(
        ((nb_cols_in + 2 * padding[1] - dilation[1] * (nb_cols_kernel - 1) - 1) / stride[1]) + 1
    )

    # Apply padding to the input
    if padding != (0, 0):
        padding_mode = "constant"
        input = torch.nn.functional.pad(
            input, (padding[1], padding[1], padding[0], padding[0]), padding_mode
        )
        # Update shape after padding
        nb_rows_in += 2 * padding[0]
        nb_cols_in += 2 * padding[1]

    # We want to get relative positions of values in the input tensor that are used
    # by one filter convolution.
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
    im_flat = input.reshape(batch_size, -1)
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
    weight_reshaped = weight.reshape(nb_channels_out // groups, -1).t()

    return (
        im_reshaped,
        weight_reshaped,
        torch.tensor(batch_size),
        torch.tensor(nb_channels_out),
        torch.tensor(nb_rows_out),
        torch.tensor(nb_cols_out),
    )


@allow_command
def _post_conv(bias, res, batch_size, nb_channels_out, nb_rows_out, nb_cols_out):
    """
    This is a block of local computation done at the end of the convolution. It
    basically reshape the matrix back to the shape it should have with a regular
    convolution.

    Because all the computation are local, we add the @allow_command and run it directly
    on each share of the additive sharing tensor, when running mpc computations
    """
    batch_size, nb_channels_out, nb_rows_out, nb_cols_out = (
        batch_size.item(),
        nb_channels_out.item(),
        nb_rows_out.item(),
        nb_cols_out.item(),
    )
    # Add a bias if needed
    if bias is not None:
        if bias.is_wrapper and res.is_wrapper:
            res += bias
        elif bias.is_wrapper:
            res += bias.child
        else:
            res += bias

    # ... And reshape it back to an image
    res = (
        res.permute(0, 2, 1)
        .reshape(batch_size, nb_channels_out, nb_rows_out, nb_cols_out)
        .contiguous()
    )

    return res


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    Overloads torch.nn.functional.conv2d to be able to use MPC on convolutional networks.
    The idea is to unroll the input and weight matrices to compute a
    matrix multiplication equivalent to the convolution.
    Args:
        input: input image
        weight: convolution kernels
        bias: optional additive bias
        stride: stride of the convolution kernels
        padding:  implicit paddings on both sides of the input.
        dilation: spacing between kernel elements
        groups: split input into groups, in_channels should be divisible by the number of groups
    Returns:
        the result of the convolution as a fixed precision tensor.
    """
    input_fp, weight_fp = input, weight

    if isinstance(input.child, FrameworkTensor) or isinstance(weight.child, FrameworkTensor):
        assert isinstance(input.child, FrameworkTensor)
        assert isinstance(weight.child, FrameworkTensor)
        im_reshaped, weight_reshaped, *params = _pre_conv(
            input, weight, bias, stride, padding, dilation, groups
        )
        if groups > 1:
            res = []
            chunks_im = torch.chunk(im_reshaped, groups, dim=2)
            chunks_weights = torch.chunk(weight_reshaped, groups, dim=0)
            for g in range(groups):
                tmp = chunks_im[g].matmul(chunks_weights[g])
                res.append(tmp)
            result = torch.cat(res, dim=2)
        else:
            result = im_reshaped.matmul(weight_reshaped)
        result = _post_conv(bias, result, *params)
        return result.wrap()

    input, weight = input.child, weight.child

    if bias is not None:
        bias = bias.child
        assert isinstance(
            bias, sy.AdditiveSharingTensor
        ), "Have you provided bias as a kwarg? If so, please remove `bias=`."

    locations = input.locations

    im_reshaped_shares = {}
    weight_reshaped_shares = {}
    params = {}
    for location in locations:
        input_share = input.child[location.id]
        weight_share = weight.child[location.id]
        bias_share = bias.child[location.id] if bias is not None else None
        (
            im_reshaped_shares[location.id],
            weight_reshaped_shares[location.id],
            *params[location.id],
        ) = remote(_pre_conv, location=location)(
            input_share,
            weight_share,
            bias_share,
            stride,
            padding,
            dilation,
            groups,
            return_value=False,
            return_arity=6,
        )

    im_reshaped = sy.FixedPrecisionTensor(**input_fp.get_class_attributes()).on(
        sy.AdditiveSharingTensor(im_reshaped_shares, **input.get_class_attributes()), wrap=False
    )
    weight_reshaped = sy.FixedPrecisionTensor(**weight_fp.get_class_attributes()).on(
        sy.AdditiveSharingTensor(weight_reshaped_shares, **input.get_class_attributes()), wrap=False
    )

    # Now that everything is set up, we can compute the convolution as a simple matmul
    if groups > 1:
        res = []
        chunks_im = torch.chunk(im_reshaped, groups, dim=2)
        chunks_weights = torch.chunk(weight_reshaped, groups, dim=0)
        for g in range(groups):
            tmp = chunks_im[g].matmul(chunks_weights[g])
            res.append(tmp)
        res_fp = torch.cat(res, dim=2)
        res = res_fp.child
    else:
        res_fp = im_reshaped.matmul(weight_reshaped)
        res = res_fp.child

    # and then we reshape the result
    res_shares = {}
    for location in locations:
        bias_share = bias.child[location.id] if bias is not None else None
        res_share = res.child[location.id]
        res_share = remote(_post_conv, location=location)(
            bias_share, res_share, *params[location.id]
        )
        res_shares[location.id] = res_share

    result_fp = sy.FixedPrecisionTensor(**res_fp.get_class_attributes()).on(
        sy.AdditiveSharingTensor(res_shares, **res.get_class_attributes()), wrap=False
    )
    return result_fp


@allow_command
def _pre_pool(input, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    """
    This is a block of local computation done at the beginning of the pool. It
    basically does the matrix unrolling to be able to do the pooling as a single
    max or average operation
    """
    original_dim = len(input.shape)
    if len(input.shape) == 3:
        input = input.reshape(1, *input.shape)
    elif len(input.shape) == 2:
        input = input.reshape(1, 1, *input.shape)

    # Change to tuple if not one
    stride = torch.nn.modules.utils._pair(stride)
    padding = torch.nn.modules.utils._pair(padding)
    dilation = torch.nn.modules.utils._pair(dilation)

    # Extract a few useful values
    batch_size, nb_channels_in, nb_rows_in, nb_cols_in = input.shape
    nb_channels_out, nb_channels_kernel, nb_rows_kernel, nb_cols_kernel = (
        nb_channels_in,
        1,
        kernel_size,
        kernel_size,
    )

    # Check if inputs are coherent
    # assert nb_channels_in == nb_channels_kernel * groups
    assert nb_channels_in % groups == 0
    assert nb_channels_out % groups == 0

    # Compute output shape
    nb_rows_out = int(
        ((nb_rows_in + 2 * padding[0] - dilation[0] * (nb_rows_kernel - 1) - 1) / stride[0]) + 1
    )
    nb_cols_out = int(
        ((nb_cols_in + 2 * padding[1] - dilation[1] * (nb_cols_kernel - 1) - 1) / stride[1]) + 1
    )

    # Apply padding to the input
    if padding != (0, 0):
        padding_mode = "constant"
        input = torch.nn.functional.pad(
            input, (padding[1], padding[1], padding[0], padding[0]), padding_mode
        )
        # Update shape after padding
        nb_rows_in += 2 * padding[0]
        nb_cols_in += 2 * padding[1]

    # We want to get relative positions of values in the input tensor that are used by
    # one filter convolution.
    # It basically is the position of the values used for the top left convolution.
    pattern_ind = []
    for r in range(nb_rows_kernel):
        for c in range(nb_cols_kernel):
            pixel = r * nb_cols_in * dilation[0] + c * dilation[1]
            pattern_ind.append(pixel)

    # The image tensor is reshaped for the matrix multiplication:
    # on each row of the new tensor will be the input values used for each filter convolution
    # We will get a matrix [[in values to compute out value 0],
    #                       [in values to compute out value 1],
    #                       ...
    #                       [in values to compute out value nb_rows_out*nb_cols_out]]
    im_flat = input.reshape(batch_size, nb_channels_in, -1)
    im_reshaped = []
    for cur_row_out in range(nb_rows_out):
        for cur_col_out in range(nb_cols_out):
            # For each new output value, we just need to shift the receptive field
            offset = cur_row_out * stride[0] * nb_cols_in + cur_col_out * stride[1]
            tmp = [ind + offset for ind in pattern_ind]
            im_reshaped.append(im_flat[:, :, tmp])
    im_reshaped = torch.stack(im_reshaped).permute(1, 2, 0, 3)

    return (
        im_reshaped,
        torch.tensor(batch_size),
        torch.tensor(nb_channels_out),
        torch.tensor(nb_rows_out),
        torch.tensor(nb_cols_out),
        torch.tensor(original_dim),
    )


@allow_command
def _post_pool(res, batch_size, nb_channels_out, nb_rows_out, nb_cols_out, original_dim):
    """
    This is a block of local computation done at the end of the pool. It reshapes
    the output to the expected shape
    """
    batch_size, nb_channels_out, nb_rows_out, nb_cols_out = (
        batch_size.item(),
        nb_channels_out.item(),
        nb_rows_out.item(),
        nb_cols_out.item(),
    )

    # ... And reshape it back to an image
    res = res.reshape(  # .permute(0, 2, 1)
        batch_size, nb_channels_out, nb_rows_out, nb_cols_out
    ).contiguous()

    if original_dim < 4:
        res = res.reshape(*res.shape[-original_dim:])

    return res


def max_pool2d(
    input,
    kernel_size: int = 2,
    stride: int = 2,
    padding=0,
    dilation=1,
    ceil_mode=None,
    return_indices=None,
):
    return _pool2d(
        input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        mode="max",
    )


def avg_pool2d(
    input,
    kernel_size: int = 2,
    stride: int = 2,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    return _pool2d(
        input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=1,
        ceil_mode=ceil_mode,
        mode="avg",
    )


def _pool2d(
    input, kernel_size: int = 2, stride: int = 2, padding=0, dilation=1, ceil_mode=None, mode="avg"
):
    if isinstance(kernel_size, tuple):
        assert kernel_size[0] == kernel_size[1]
        kernel_size = kernel_size[0]
    if isinstance(stride, tuple):
        assert stride[0] == stride[1]
        stride = stride[0]

    input_fp = input
    input = input.child

    locations = input.locations

    im_reshaped_shares = {}
    params = {}
    for location in locations:
        input_share = input.child[location.id]
        im_reshaped_shares[location.id], *params[location.id] = remote(
            _pre_pool, location=location
        )(input_share, kernel_size, stride, padding, dilation, return_value=False, return_arity=6)

    im_reshaped = sy.AdditiveSharingTensor(im_reshaped_shares, **input.get_class_attributes())

    if mode == "max":
        # We have optimisations when the kernel is small, namely a square of size 2 or 3
        # to reduce the number of rounds and the total number of comparisons.
        # See more in Appendice C.3 https://arxiv.org/pdf/2006.04593.pdf
        def max_half_split(tensor4d, half_size):
            """
            Split the tensor on 2 halves on the last dim and return the maximum half
            """
            left, right = tensor4d[:, :, :, :half_size], tensor4d[:, :, :, half_size:]
            max_half = left + (right >= left) * (right - left)
            return max_half

        if im_reshaped.shape[-1] == 4:
            # Compute the max as a binary tree: 2 steps are needed for 4 values
            res = max_half_split(im_reshaped, 2)
            res = max_half_split(res, 1)
        elif im_reshaped.shape[-1] == 9:
            # For 9 values we need 4 steps: we process the 8 first values and then
            # compute the max with the 9th value
            res = max_half_split(im_reshaped[:, :, :, :8], 4)
            res = max_half_split(res, 2)
            left = max_half_split(res, 1)
            right = im_reshaped[:, :, :, 8:]
            res = left + (right >= left) * (right - left)
        else:
            res = im_reshaped.max(dim=-1)
    elif mode == "avg":
        res = im_reshaped.mean(dim=-1)
    else:
        raise ValueError(f"In pool2d, mode should be avg or max, not {mode}.")

    res_shares = {}
    for location in locations:
        res_share = res.child[location.id]
        res_share = remote(_post_pool, location=location)(res_share, *params[location.id])
        res_shares[location.id] = res_share

    result_fp = sy.FixedPrecisionTensor(**input_fp.get_class_attributes()).on(
        sy.AdditiveSharingTensor(res_shares, **res.get_class_attributes()), wrap=False
    )
    return result_fp


def adaptive_avg_pool2d(tensor, output_size):
    if isinstance(output_size, tuple):
        assert output_size[0] == output_size[1]
        output_size = output_size[0]
    assert tensor.shape[2] == tensor.shape[3]
    input_size = tensor.shape[2]
    assert input_size >= output_size
    stride = input_size // output_size
    kernel_size = input_size - (output_size - 1) * stride
    padding = 0
    return avg_pool2d(tensor, kernel_size, stride, padding)
