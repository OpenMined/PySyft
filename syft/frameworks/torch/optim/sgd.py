class SGD:
    def __init__(self, params, lr):

        # doing this as a list isn't memory efficient
        self.params = list(params)
        self.lr = lr

    def step(self, batch_size):

        # TODO: all all the SGD features from PyTorch's SGD
        for p in self.params:
            p.data.sub_(p.grad * self.lr / batch_size)

    def zero_grad(self):
        """We need to use a try/catch because our pointers
        don't always know whether they point to an object
        that exists or not. At the moment, finding out requires
        calling .get() which compromises privacy. There may be
        a more nuanced way to do this in the future."""

        for p in self.params:
            try:
                p.grad -= p.grad
            except:
                ""
