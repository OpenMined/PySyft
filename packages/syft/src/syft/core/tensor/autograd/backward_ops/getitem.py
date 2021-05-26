# stdlib
import uuid

# syft relative
from ..tensor import AutogradTensor
from .op import Op


class GetItemOp(Op):
    def forward(self, x: AutogradTensor, y: AutogradTensor) -> AutogradTensor:
        self.x = x
        self.y = y

    def _backward(self, grad: AutogradTensor, backprop_id: uuid.uuid4):
        if self.x.requires_grad:
            pass

        if self.y.requires_grad:
            pass
