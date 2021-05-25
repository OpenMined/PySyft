# syft relative
from ..tensor import AutogradTensor
from .op import Op


class RepeatOp(Op):
    """Repeat operation across a dimension"""

    def forward(self, x: AutogradTensor, repeats, axis=None):
        self.x = x
        self.repeats = repeats
        self.axis = axis

        self.input_shape = self.x.shape

        output = x.child.repeat(repeats=repeats, axis=axis)

        self.output_shape = output.shape

        return AutogradTensor(output, requires_grad=x.requires_grad)

    def _backward(self, grad, backprop_id):

        if self.x.requires_grad:

            requires_grad = grad.requires_grad

            axis = self.axis
            if axis is None:
                axis = len(self.input_shape) - 1

            intermediate_shape = list(self.input_shape)
            intermediate_shape.insert(axis + 1, -1)

            grad = grad.child.reshape(intermediate_shape)

            grad = grad.sum(axis=axis + 1)

            self.x.add_grad(AutogradTensor(grad, requires_grad=requires_grad))

            if self.x.grad_fn:
                self.x.backward(backprop_id=backprop_id)
