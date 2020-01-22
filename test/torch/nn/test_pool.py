import syft.frameworks.torch.nn as nn2
import torch as th
import torch.nn as nn


def test_pool2d():
    """
    Test the Pool2d module to ensure that it produces the exact same
    output as the primary torch implementation, in the same order.
    """

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

    pool = nn.AvgPool2d(2)
    pool2 = nn2.AvgPool2d(2)

    data = th.rand(10, 1, 8, 8)

    model_out = model(data)
    out = pool(model_out)
    out2 = pool2(model_out)

    assert th.eq(out, out2).all()
