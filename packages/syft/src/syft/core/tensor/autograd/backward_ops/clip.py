# stdlib
import uuid

# syft relative
from ..tensor import AutogradTensor
from .op import Op


class ClipOp(Op):
    def forward(
        self, x: AutogradTensor, y: AutogradTensor, z: AutogradTensor
    ) -> AutogradTensor:
        self.x = x
        self.y = y
        self.z = z

    def _backward(self, grad: AutogradTensor, backprop_id: uuid.UUID):
        if self.x.requires_grad:
            pass

        if self.y.requires_grad:
            pass

        if self.z.requires_grad:
            pass
