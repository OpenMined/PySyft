class SumOp(Op):
    '''Sum operation across a dimension'''

    def forward(self, x: AutogradTensor, axis):
        self.x = x
        self.axis = axis
        self.dim_at_axis = self.x.shape[self.axis]
        self.backward_shape = self.x.shape

        return AutogradTensor(x.child.sum(axis), requires_grad=x.requires_grad)

    def _backward(self, grad, backprop_id):

        if self.x.requires_grad:

            requires_grad = grad.requires_grad

            grad = np.expand_dims(grad.child, self.axis)

            grad = grad.repeat(self.dim_at_axis, axis=self.axis)

            self.x.add_grad(AutogradTensor(grad, requires_grad=requires_grad))

            if self.x.grad_fn:
                self.x.backward(backprop_id=backprop_id)