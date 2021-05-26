# stdlib
import uuid

# syft relative
from ..tensor import AutogradTensor
from .op import Op


class ResizeOp(Op):
    def forward(self, x: AutogradTensor) -> AutogradTensor:
        self.x = x

    def _backward(self, grad: AutogradTensor, backprop_id: uuid.uuid4):
        if self.x.requires_grad:
            pass
