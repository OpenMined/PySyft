class ReshapeOp(Op):
    '''Multiplication operation with 2 tensors'''

    def forward(self, x: AutogradTensor, *shape):
        self.x = x
        self.shape = shape
        self.backward_shape = self.x.shape

        return AutogradTensor(x.child.reshape(*shape), requires_grad=x.requires_grad)

    def _backward(self, grad, backprop_id):

        if self.x.requires_grad:

            self.x.add_grad(AutogradTensor(grad.child.reshape(self.backward_shape)))

            if self.x.grad_fn:
                self.x.backward(backprop_id=backprop_id)
