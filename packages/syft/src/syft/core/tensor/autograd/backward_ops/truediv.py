# stdlib
import uuid

# relative
from ..tensor import AutogradTensor
from .op import Op


class TrueDivOp(Op):
    def forward(self, x: AutogradTensor, y: AutogradTensor) -> AutogradTensor:  # type: ignore
        self.x = x
        self.y = y

        # This is just a placeholder to remove return type linting errors until this method is built out fully
        return AutogradTensor(x.child / y)

    def _backward(self, grad: AutogradTensor, backprop_id: uuid.UUID) -> None:
        if self.x.requires_grad:
            pass

        if self.y.requires_grad:
            pass
