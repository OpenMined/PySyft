# stdlib
import uuid

# relative
from ..tensor import AutogradTensor
from .op import Op


class InvertOp(Op):
    def forward(self, x: AutogradTensor) -> AutogradTensor:  # type: ignore
        self.x = x

        # This is just a placeholder to suppress linting errors until the method is built out
        return x

    def _backward(self, grad: AutogradTensor, backprop_id: uuid.UUID) -> None:
        if self.x.requires_grad:
            pass
