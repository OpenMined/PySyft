# syft relative
from ..tensor import AutogradTensor
from .op import Op
import numpy as np


class SumOp(Op):
    '''Sum operation across a dimension'''

    def forward(self, x: AutogradTensor, axis):
        self.x = x
        self.axis = axis
        if axis is not None:
            # obj.sum() can be called without dims
            self.dim_at_axis = self.x.shape[self.axis]
        else:
            self.dim_at_axis = None
        self.backward_shape = self.x.shape

        result = x.child.sum(axis)

        return AutogradTensor(result, requires_grad=x.requires_grad)

    def _backward(self, grad, backprop_id):

        if self.x.requires_grad:

            requires_grad = grad.requires_grad

            grad = np.expand_dims(grad.child, self.axis)

            grad = grad.repeat(self.dim_at_axis, axis=self.axis)

            self.x.add_grad(AutogradTensor(grad, requires_grad=requires_grad))

            if self.x.grad_fn:
                self.x.backward(backprop_id=backprop_id)
