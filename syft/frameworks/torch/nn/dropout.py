from torch.nn import Module
from . import functional as F


class _DropoutNd(Module):
    __constants__ = ["p", "inplace"]

    def __init__(self, p=0.5, inplace=False):
        super(_DropoutNd, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.inplace = inplace

    def extra_repr(self):
        return f"p={self.p}, inplace={self.inplace}"


class Dropout(_DropoutNd):
    """
    This class tries to be an exact python port of the torch.nn.Dropout
    module. Because PySyft cannot hook into layers which are implemented in C++,
    our special functionalities (such as encrypted computation) do not work with
    torch.nn.Dropout and so we must have python ports available for all layer types
    which we seek to use.

    Note: This module is tested to ensure that it outputs the exact output
    values that the main module outputs in the same order that the main module does.

    This module has not yet been tested with GPUs but should work out of the box.

    Args:
        p (float, optional): probability of an element to be zero-ed.
        inplace (bool, optional): If set to ``True``, will do this operation in-place

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input
    """

    def forward(self, input):
        return F.dropout(input, self.p, self.training, self.inplace)


class Dropout2d(_DropoutNd):
    """
    Randomly zero out entire channels (a channel is a 2D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 2D tensor :math:`\text{input}[i, j]`).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    Usually the input comes from :class:`nn.Conv2d` modules.

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`.

    Args:
        p (float, optional): probability of an element to be zero-ed.
        inplace (bool, optional): If set to ``True``, will do this operation in-place

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    """

    def forward(self, input):
        return F.dropout2d(input, self.p, self.training, self.inplace)
