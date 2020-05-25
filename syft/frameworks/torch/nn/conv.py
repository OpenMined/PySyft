import torch as th
import torch.nn as nn

from syft.frameworks.torch.nn.functional import conv2d


class Conv2d(nn.Module):
    """
    This class tries to be an exact python port of the torch.nn.Conv2d
    module. Because PySyft cannot hook into layers which are implemented in C++,
    our special functionalities (such as encrypted computation) do not work with
    torch.nn.Conv2d and so we must have python ports available for all layer types
    which we seek to use.

    Note: This module is tested to ensure that it outputs the exact output
    values that the main module outputs in the same order that the main module does.

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

        self.weight = th.Tensor(temp_init.weight).fix_prec()
        if bias:
            self.bias = th.Tensor(temp_init.bias).fix_prec()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # These are modified and converted to tuples
        self.stride = temp_init.stride
        self.padding = temp_init.padding
        self.dilation = temp_init.dilation

        self.groups = groups
        self.padding_mode = padding_mode

    def forward(self, input):

        assert input.shape[1] == self.in_channels

        return conv2d(
            input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


# IMPLEMENTED BY @IAMTRASK IN https://github.com/OpenMined/PySyft/pull/2896

# class Conv2d(Module):
#     """
#     This class is the beginning of an exact python port of the torch.nn.Conv2d
#     module. Because PySyft cannot hook into layers which are implemented in C++,
#     our special functionalities (such as encrypted computation) do not work with
#     torch.nn.Conv2d and so we must have python ports available for all layer types
#     which we seek to use.
#
#     Note that this module has been tested to ensure that it outputs the exact output
#     values that the main module outputs in the same order that the main module does.
#
#     However, there is often some rounding error of unknown origin, usually less than
#     1e-6 in magnitude.
#
#     This module has not yet been tested with GPUs but should work out of the box.
#     """
#
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size,
#         stride=1,
#         padding=0,
#         dilation=1,
#         groups=1,
#         bias=False,
#         padding_mode="zeros",
#     ):
#         """For information on the constructor arguments, please see PyTorch's
#         documentation in torch.nn.Conv2d"""
#
#         super().__init__()
#
#         # because my particular experiment does not demand full functionality of
#         # a convolutional layer, I will only implement the basic functionality.
#         # These assertions are the required settings.
#
#         assert in_channels == 1
#         assert stride == 1
#         assert padding == 0
#         assert dilation == 1
#         assert groups == 1
#         assert padding_mode == "zeros"
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups
#         self.has_bias = bias
#         self.padding_mode = padding_mode
#
#         temp_init = th.nn.Conv2d(
#             in_channels=self.in_channels,
#             out_channels=self.out_channels,
#             kernel_size=self.kernel_size,
#             stride=self.stride,
#             padding=self.padding,
#             dilation=self.dilation,
#             groups=self.groups,
#             bias=self.has_bias,
#             padding_mode=self.padding_mode,
#         )
#
#         self.weight = temp_init.weight
#         self.bias = temp_init.bias
#
#     def forward(self, data):
#
#         batch_size, _, rows, cols = data.shape
#
#         flattened_model = self.weight.reshape(self.out_channels, -1)
#         flattened_data = th.nn.functional.unfold(data, kernel_size=self.kernel_size)
#
#         # Loop over batch as direct multiplication results in rounding errors
#         kernel_results = list()
#         for n in range(0, batch_size):
#             kernel_results.append(flattened_model @ flattened_data[n])
#
#         pred = th.stack(kernel_results, axis=0).view(
#             batch_size, self.out_channels,
#             rows - self.kernel_size + 1, cols - self.kernel_size + 1
#         )
#
#         if self.has_bias:
#             pred = pred + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(
#                 batch_size,
#                 self.out_channels,
#                 rows - self.kernel_size + 1,
#                 cols - self.kernel_size + 1,
#             )
#
#         return pred
