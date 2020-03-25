import torch as th
import torch.nn as nn


class Conv2d(nn.Module):
    """
    This class tries to be an exact python port of the torch.nn.Conv2d
    module. Because PySyft cannot hook into layers which are implemented in C++,
    our special functionalities (such as encrypted computation) do not work with
    torch.nn.Conv2d and so we must have python ports available for all layer types
    which we seek to use.

    TODO: This module NEEDS TO BE tested to ensure that it outputs the exact output
    values that the main module outputs in the same order that the main module does.

    @iamtrask says: There is often some rounding error of unknown origin, usually less than
    1e-6 in magnitude.

    This module has not yet been tested with GPUs but should work out of the box.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        padding_mode="zeros",
    ):
        """For information on the constructor arguments, please see PyTorch's
        documentation in torch.nn.Conv2d"""

        super().__init__()

        # temp_init to get weights and bias
        temp_init = th.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        # How to initialize the weights and bias ?
        # Over here, we are returned torch tensors with the right shapes
        # But do we need torch tensors ?
        self.weight = temp_init.weight
        self.bias = temp_init.bias

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # These are modified and converted to tuples
        self.stride = temp_init.stride
        self.padding = temp_init.padding
        self.dilation = temp_init.dilation

        self.groups = groups
        self.padding_mode = padding_mode

    def forward(self, x):

        # Assert right input shape
        assert input.shape[1] == self.in_channels

        return self.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.padding_mode,
        )

    # TODO: This can be moved to nn.functional similar to the implementation in torch
    def conv2d(self, input, weight, bias, stride, padding, dilation, groups, padding_mode):
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

        # Extract a few useful values
        # Note: Unlike pytorch only batch_first is supported
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
            padding_mode = "constant" if padding_mode == "zeros" else padding_mode
            input = nn.functional.pad(
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
        im_reshaped = th.stack(im_reshaped).permute(1, 0, 2)

        # The convolution kernels are also reshaped for the matrix multiplication
        # We will get a matrix [[weights for out channel 0],
        #                       [weights for out channel 1],
        #                       ...
        #                       [weights for out channel nb_channels_out]].TRANSPOSE()
        weight_reshaped = weight.view(nb_channels_out // groups, -1).t()

        # Now that everything is set up, we can compute the result
        if groups > 1:
            res = []
            chunks_im = th.chunk(im_reshaped, groups, dim=2)
            chunks_weights = th.chunk(weight_reshaped, groups, dim=0)
            for g in range(groups):
                tmp = chunks_im[g].matmul(chunks_weights[g])
                res.append(tmp)
            res = th.cat(res, dim=2)
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
