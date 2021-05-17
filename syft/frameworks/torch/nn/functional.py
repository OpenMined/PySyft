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
        if exponential_average_factor != 0:
            raise NotImplementedError(
                "exponential_average_factor is not supported for the moment and should be set to 0"
            )
        mean = input.mean(dim=0)
        var = input.var(dim=0) + eps
        sqrt_inv_var = var.reciprocal(method="newton")
    else:
        mean = running_mean
        var = running_var
        sqrt_inv_var = var  # already done in module.encrypt() !!

    normalized = sqrt_inv_var * (input - mean)
    result = normalized * weight + bias

    result = result.t()
    result = result.reshape(*input_shape)
    result = result.permute(1, 0, 2, 3)

    return result


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
    if nb_channels_in % groups != 0:
        raise ValueError(
            f"Given inputs are not supported. Given inputs: "
            f"nb_channels_in: {nb_channels_in}, groups: {groups}"
        )
    if nb_channels_out % groups != 0:
        raise ValueError(
            f"Given inputs are not supported. Given inputs: "
            f"nb_channels_out: {nb_channels_out}, groups: {groups}"
        )

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
def _post_pool(res, indices, batch_size, nb_channels_out, nb_rows_out, nb_cols_out, original_dim):
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

    if indices is not None:
        indices = indices.reshape(
            batch_size, nb_channels_out, nb_rows_out, nb_cols_out, *indices.shape[-2:]
        ).contiguous()

    if original_dim < 4:
        res = res.reshape(*res.shape[-original_dim:])

        if indices is not None:
            raise NotImplementedError()

    if indices is not None:
        return res, indices
    else:
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
        return_indices=return_indices,
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
    input,
    kernel_size: int = 2,
    stride: int = 2,
    padding=0,
    dilation=1,
    ceil_mode=None,
    mode="avg",
    return_indices=False,
):
    if isinstance(kernel_size, tuple):
        if kernel_size[0] != kernel_size[1]:
            raise ValueError(
                f"kernel_size[0] should be equal to kernel_size[1], "
                f"Check the given kernel_size {kernel_size}"
            )
        kernel_size = kernel_size[0]
    if isinstance(stride, tuple):
        if stride[0] != stride[1]:
            raise ValueError(
                f"stride[0] should be equal to stride[1], " f"Check the given stride {stride}"
            )
        stride = stride[0]

    input_fp = input

    if isinstance(input_fp.child, FrameworkTensor):
        im_reshaped, *params = _pre_pool(input.child, kernel_size, stride, padding, dilation)

    else:
        input = input.child

        locations = input.locations

        im_reshaped_shares = {}
        params = {}
        for location in locations:
            input_share = input.child[location.id]
            im_reshaped_shares[location.id], *params[location.id] = remote(
                _pre_pool, location=location
            )(
                input_share,
                kernel_size,
                stride,
                padding,
                dilation,
                return_value=False,
                return_arity=6,
            )

        im_reshaped = sy.AdditiveSharingTensor(im_reshaped_shares, **input.get_class_attributes())

    if mode == "max":
        # We have optimisations when the kernel is small, namely a square of size 2 or 3
        # to reduce the number of rounds and the total number of comparisons.
        # See more in Appendice C.3 https://arxiv.org/pdf/2006.04593.pdf
        def max_half_split(tensor4d, half_size):
            """
            Split the tensor on 2 halves on the last dim and compare them

            Return
                - the maximum half
                - the index of the elements that made the maximum between the left and right halves
            """
            left, right = tensor4d[:, :, :, :half_size], tensor4d[:, :, :, half_size:]

            max_half_boolean = right >= left
            max_half_value = left + max_half_boolean * (right - left)

            return max_half_value, max_half_boolean

        if im_reshaped.shape[-1] == 4:
            # Compute the max as a binary tree: 2 steps are needed for 4 values
            res, idx2 = max_half_split(im_reshaped, 2)
            res, idx1 = max_half_split(res, 1)

            if return_indices:
                idx2_0 = idx2[:, :, :, :1]
                idx2_1 = idx2[:, :, :, 1:]
                if isinstance(idx2_0, torch.BoolTensor):
                    indices = torch.stack(
                        [
                            (~idx2_0) * (~idx1),
                            (~idx2_1) * idx1,
                            idx2_0 * (~idx1),
                            idx2_1 * idx1,
                        ],
                        dim=-1,
                    )
                else:
                    indices = torch.stack(
                        [
                            (1 - idx2_0) * (1 - idx1),
                            (1 - idx2_1) * idx1,
                            idx2_0 * (1 - idx1),
                            idx2_1 * idx1,
                        ],
                        dim=-1,
                    )
                assert isinstance(kernel_size, int)
                indices = indices.reshape(*res.shape, kernel_size, kernel_size)

        elif im_reshaped.shape[-1] == 9:
            # For 9 values we need 4 steps: we process the 8 first values and then
            # compute the max with the 9th value
            res, idx4 = max_half_split(im_reshaped[:, :, :, :8], 4)
            res, idx2 = max_half_split(res, 2)
            left, idx1 = max_half_split(res, 1)

            if return_indices:
                raise NotImplementedError("Not implemented for the moment")

            right = im_reshaped[:, :, :, 8:]
            right_ge_left = right >= left
            res = left + right_ge_left * (right - left)
        else:
            if return_indices:
                raise NotImplementedError("Not implemented for the moment")
            if return_indices:
                res, indices = im_reshaped.max(dim=-1, return_indices=return_indices)
            else:
                res = im_reshaped.max(dim=-1, return_indices=return_indices)

    elif mode == "avg":
        if isinstance(input_fp.child, FrameworkTensor):
            sum_value = im_reshaped.sum(dim=-1)
            m = im_reshaped.numel() // sum_value.numel()
            res = sum_value // m
        else:
            res = im_reshaped.mean(dim=-1)
    else:
        raise ValueError(f"In pool2d, mode should be avg or max, not {mode}.")

    if isinstance(input_fp.child, FrameworkTensor):
        if return_indices:
            result, indices = _post_pool(res, indices, *params)
        else:
            result = _post_pool(res, None, *params)
        result_fp = sy.FixedPrecisionTensor(**input_fp.get_class_attributes()).on(
            result, wrap=False
        )
        if return_indices:
            indices_fp = sy.FixedPrecisionTensor(**input_fp.get_class_attributes()).on(
                indices, wrap=False
            )
            return result_fp, indices_fp
        else:
            return result_fp
    else:
        res_shares = {}
        indices_shares = {}
        for location in locations:
            res_share = res.child[location.id]
            if return_indices:
                indices_share = indices.child[location.id]
                res_share, indices_share = remote(_post_pool, location=location)(
                    res_share, indices_share, *params[location.id], return_arity=2
                )
                res_shares[location.id] = res_share
                indices_shares[location.id] = indices_share
            else:
                res_share = remote(_post_pool, location=location)(
                    res_share, None, *params[location.id]
                )
                res_shares[location.id] = res_share

        result_fp = sy.FixedPrecisionTensor(**input_fp.get_class_attributes()).on(
            sy.AdditiveSharingTensor(res_shares, **res.get_class_attributes()), wrap=False
        )
        if return_indices:
            indices_shares = sy.AdditiveSharingTensor(indices_shares, **res.get_class_attributes())
            indices_fp = sy.FixedPrecisionTensor(**input_fp.get_class_attributes()).on(
                indices_shares, wrap=False
            )
            return result_fp, indices_fp
        else:
            return result_fp


def _max_pool2d_backward(
    self,
    indices,
    kernel_size,
    padding=None,
    stride=None,
    dilation=1,
    ceil_mode=False,
    output_size=None,
):
    """Implements the backwards for a `max_pool2d` call."""
    # Setup padding
    if padding is None:
        padding = 0
    if isinstance(padding, int):
        padding = padding, padding
    assert isinstance(padding, tuple), "padding must be a int, tuple, or None"
    p0, p1 = padding

    # Setup stride
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = stride, stride
    assert isinstance(stride, tuple), "stride must be a int, tuple, or None"
    s0, s1 = stride

    # Setup dilation
    if isinstance(stride, int):
        dilation = dilation, dilation
    assert isinstance(dilation, tuple), "dilation must be a int, tuple, or None"
    d0, d1 = dilation

    # Setup kernel_size
    if isinstance(kernel_size, int):
        kernel_size = kernel_size, kernel_size
    assert isinstance(padding, tuple), "padding must be a int or tuple"
    k0, k1 = kernel_size

    assert self.dim() == 4, "Input to _max_pool2d_backward must have 4 dimensions"
    assert indices.dim() == 6, "Indices input for _max_pool2d_backward must have 6 dimensions"

    # Computes one-hot gradient blocks from each output variable that
    # has non-zero value corresponding to the argmax of the corresponding
    # block of the max_pool2d input.
    kernels = self.reshape(self.shape + (1, 1)) * indices

    # Use minimal size if output_size is not specified.
    if output_size is None:
        output_size = (
            self.shape[0],
            self.shape[1],
            s0 * self.shape[2] - 2 * p0,
            s1 * self.shape[3] - 2 * p1,
        )

    # Account for input padding
    result_size = list(output_size)
    result_size[-2] += 2 * p0
    result_size[-1] += 2 * p1

    # Account for input padding implied by ceil_mode
    if ceil_mode:
        c0 = self.shape[-1] * s1 + (k1 - 1) * d1 - output_size[-1]
        c1 = self.shape[-2] * s0 + (k0 - 1) * d0 - output_size[-2]
        result_size[-2] += c0
        result_size[-1] += c1

    # Sum the one-hot gradient blocks at corresponding index locations.
    if isinstance(self.child, FrameworkTensor):
        result = torch.zeros(result_size).fix_prec(**self.get_class_attributes()).child
    else:
        result = (
            torch.zeros(result_size)
            .fix_prec(**self.get_class_attributes())
            .share(*self.child.locations, **self.child.get_class_attributes())
            .child
        )
    for i in range(self.shape[2]):
        for j in range(self.shape[3]):
            left_ind = s0 * i
            top_ind = s1 * j

            result[
                :,
                :,
                left_ind : left_ind + k0 * d0 : d0,
                top_ind : top_ind + k1 * d1 : d1,
            ] += kernels[:, :, i, j]

    # Remove input padding
    if ceil_mode:
        result = result[:, :, : result.shape[2] - c0, : result.shape[3] - c1]
    result = result[:, :, p0 : result.shape[2] - p0, p1 : result.shape[3] - p1]

    return result


def adaptive_avg_pool2d(tensor, output_size):
    if isinstance(output_size, tuple):
        if output_size[0] != output_size[1]:
            raise ValueError("Check given output_size")
        output_size = output_size[0]

    if tensor.shape[2] != tensor.shape[3]:
        raise ValueError(
            f"Shape of given tensor is invalid, "
            f"tensor.shape[2] should be equal to tensor.shape[3], "
            f"Given tensor.shape: {tensor.shape}"
        )

    input_size = tensor.shape[2]

    if input_size < output_size:
        raise ValueError("tensor.shape[2] should be greater or equal to output_size")

    stride = input_size // output_size
    kernel_size = input_size - (output_size - 1) * stride
    padding = 0
    return avg_pool2d(tensor, kernel_size, stride, padding)
