# stdlib
import uuid

# relative
from ..tensor import AutogradTensor
from .op import Op


class CumSumOp(Op):
    def forward(self, x: AutogradTensor, y: AutogradTensor) -> None:  # type: ignore
        self.x = x
        self.y = y

    def _backward(self, grad: AutogradTensor, backprop_id: uuid.UUID) -> None:
        if self.x.requires_grad:
            pass

        if self.y.requires_grad:
            pass
