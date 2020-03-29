import syft.frameworks.torch.nn as nn2
import torch as th
import torch.nn as nn
import pytest


def test_conv2d():
    """
    Test the Conv2d module to ensure that it produces the exact same
    output as the primary torch implementation, in the same order.
    """

    # Disable mkldnn to avoid rounding errors due to difference in implementation
    mkldnn_enabled_init = th._C._get_mkldnn_enabled()
    th._C._set_mkldnn_enabled(False)

    model2 = nn2.Conv2d(1, 32, 3, bias=True)

    model = nn.Conv2d(
        in_channels=1,
        out_channels=32,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    )

    model2.weight = model.weight
    model2.bias = model.bias

    data = th.rand(10, 1, 28, 28)

    out = model(data)

    out2 = model2(data)

    # Reset mkldnn to the original state
    th._C._set_mkldnn_enabled(mkldnn_enabled_init)

    assert th.eq(out, out2).all()
