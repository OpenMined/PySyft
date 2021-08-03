# stdlib
import uuid

# relative
from ..tensor import AutogradTensor
from .op import Op


class DotOp(Op):
    def forward(self, x: AutogradTensor, y: AutogradTensor) -> AutogradTensor:  # type: ignore
        self.x = x
        self.y = y

        # This is just a placeholder to remove return type linting errors until the method is built
        return AutogradTensor(x.child)

    def _backward(self, grad: AutogradTensor, backprop_id: uuid.UUID) -> None:
        if self.x.requires_grad:
            pass

        if self.y.requires_grad:
            pass
