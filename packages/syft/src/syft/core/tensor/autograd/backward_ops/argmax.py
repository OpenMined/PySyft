# stdlib
import uuid

# syft relative
from ..tensor import AutogradTensor
from .op import Op


class ArgMaxOp(Op):
    def forward(self, x: AutogradTensor, axis=None) -> AutogradTensor:
        self.x = x

    def _backward(self, grad: AutogradTensor, backprop_id: uuid.UUID):
        if self.x.requires_grad:
            pass
