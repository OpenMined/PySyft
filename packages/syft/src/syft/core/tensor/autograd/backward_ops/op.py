# syft relative
from ..tensor import AutogradTensor


class Op:
    def __init__(self):
        self.backprop_id = None

    def forward(self):
        raise NotImplementedError

    def _backward(self, grad, backprop_id):
        raise NotImplementedError

    def backward(self, grad, backprop_id):

        self.backprop_id = backprop_id

        for t in self.parent_tensors:
            t.backprop_id = backprop_id

        return self._backward(grad=grad, backprop_id=backprop_id)

    def __call__(self, *args, **kwargs):

        self.parent_tensors = list()

        for arg in args:
            if isinstance(arg, AutogradTensor):
                arg.ops.append(self)
                self.parent_tensors.append(arg)

        for key, arg in kwargs.items():
            if isinstance(arg, AutogradTensor):
                arg.ops.append(self)
                self.parent_tensor.append(arg)

        self.out = self.forward(*args, **kwargs)
        self.out._grad_fn = self

        return self.out
