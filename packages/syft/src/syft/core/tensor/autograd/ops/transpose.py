# syft relative
from ..tensor import AutogradTensor
from .op import Op


class TransposeOp(Op):
    '''Repeat operation across a dimension'''

    def forward(self, x: AutogradTensor, *dims):
        self.x = x
        self.dims = dims

        reverse_t_dims = {}
        for i, d in enumerate(self.dims):
            reverse_t_dims[d] = i

        l = sorted([(x[0], x[1]) for x in reverse_t_dims.items()], key=lambda x: x[0])
        self.reverse_dims = [x[1] for x in l]

        return AutogradTensor(x.child.transpose(*dims), requires_grad=x.requires_grad)

    def _backward(self, grad, backprop_id):

        if self.x.requires_grad:

            requires_grad = grad.requires_grad

            grad = grad.child.transpose(*self.reverse_dims)

            self.x.add_grad(AutogradTensor(grad, requires_grad=requires_grad))

            if self.x.grad_fn:
                self.x.backward(backprop_id=backprop_id)
