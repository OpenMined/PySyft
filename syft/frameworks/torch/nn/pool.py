import torch as th
from torch.nn import Module
import syft as sy


class AvgPool2d(Module):
    """
    This class is the beginning of an exact python port of the torch.nn.AvgPool2d
    module. Because PySyft cannot hook into layers which are implemented in C++,
    our special functionalities (such as encrypted computation) do not work with
    torch.nn.AvgPool2d and so we must have python ports available for all layer types
    which we seek to use.

    Note that this module has been tested to ensure that it outputs the exact output
    values that the main module outputs in the same order that the main module does.

    However, there is often some rounding error of unknown origin, usually less than
    1e-6 in magnitude.

    This module has not yet been tested with GPUs but should work out of the box.
    """

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    ):
        """For information on the constructor arguments, please see PyTorch's
        documentation in torch.nn.AvgPool2d"""

        super().__init__()

        # I have not implemented all functionality from torch.nn.AvgPool2d
        # These assertions are the required settings.

        assert padding == 0
        assert ceil_mode == False
        assert count_include_pad == True
        assert divisor_override == None

        if stride is None:
            stride = kernel_size

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

        self._one_over_kernel_size = 1 / (self.kernel_size * self.kernel_size)

    def forward(self, data):

        batch_size, out_channels, rows, cols = data.shape

        kernel_results = list()

        for i in range(0, rows - self.kernel_size + 1, self.stride):
            for j in range(0, cols - self.kernel_size + 1, self.stride):
                kernel_out = (
                    data[:, :, i : i + self.kernel_size, j : j + self.kernel_size].sum((2, 3))
                    * self._one_over_kernel_size
                )
                kernel_results.append(kernel_out.unsqueeze(2))

        pred = th.cat(kernel_results, axis=2).view(
            batch_size, out_channels, int(rows / self.stride), int(cols / self.stride)
        )

        return pred

    def __repr__(self):
        return str(self)

    def __str__(self):
        out = "AvgPool2d-Handcrafted("
        out += "kernel_size=" + str(self.kernel_size) + ", "
        out += "stride=" + str(self.stride) + ", "
        out += "padding=" + str(self.padding)
        out += ")"
        return out

    def torchcraft(self):
        """Converts this handcrafted module into a torch.nn.AvgPool2d module wherein all the
        module's features are executing in C++. This will increase performance at the cost of
        some of PySyft's more advanced features such as encrypted computation."""

        kwargs = {}
        kwargs["kernel_size"] = self.kernel_size
        kwargs["stride"] = self.stride
        kwargs["padding"] = self.padding
        kwargs["ceil_mode"] = self.ceil_mode
        kwargs["count_include_pad"] = self.count_include_pad
        kwargs["divisor_override"] = self.divisor_override

        return th.nn.AvgPool2d(**kwargs)


def handcraft(self):
    """Converts a torch.nn.AvgPool2d module to a handcrafted one wherein all the
    module's features are executing in python. This is necessary for some of PySyft's
    more advanced features (like encrypted computation)."""

    kwargs = {}
    kwargs["kernel_size"] = self.kernel_size
    kwargs["stride"] = self.stride
    kwargs["padding"] = self.padding
    kwargs["ceil_mode"] = self.ceil_mode
    kwargs["count_include_pad"] = self.count_include_pad
    kwargs["divisor_override"] = self.divisor_override

    return sy.frameworks.torch.nn.AvgPool2d(**kwargs)


th.nn.AvgPool2d.handcraft = handcraft
