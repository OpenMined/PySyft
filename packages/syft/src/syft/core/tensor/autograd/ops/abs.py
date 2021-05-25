# stdlib
import uuid

# third party
import numpy as np

# syft relative
from ...passthrough import is_acceptable_simple_type
from ..tensor import AutogradTensor
from .op import Op


class AbsOp(Op):
    def forward(self, x: AutogradTensor) -> AutogradTensor:
        self.x = x

    def _backward(self, grad: AutogradTensor, backprop_id: uuid.uuid4):
        if self.x.requires_grad:
            pass
