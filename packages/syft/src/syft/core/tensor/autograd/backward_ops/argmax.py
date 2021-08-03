# stdlib
from typing import Optional
import uuid

# relative
from ..tensor import AutogradTensor
from .op import Op


class ArgMaxOp(Op):
    def forward(self, x: AutogradTensor, axis: Optional[int] = None) -> AutogradTensor:  # type: ignore
        self.x = x

        # This is just a placeholder to suppress linting errors until the method is built out
        return x.max() if not axis else x.max(axis)

    def _backward(self, grad: AutogradTensor, backprop_id: uuid.UUID) -> None:
        if self.x.requires_grad:
            pass
