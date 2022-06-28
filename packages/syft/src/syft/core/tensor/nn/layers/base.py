# relative
from ...autodp.phi_tensor import PhiTensor


class Layer:
    """
    Subclassed when implementing new types of layers.

    Each layer can keep track of the layer(s) feeding into it, a
    network's output :class:`Layer` instance can double as a handle to the full
    network.
    """

    first_layer = False

    def forward(self, input: PhiTensor, *args, **kwargs):
        raise NotImplementedError

    def backward(self, pre_grad, *args, **kwargs):
        raise NotImplementedError

    def connect_to(self, prev_layer):
        raise NotImplementedError

    @property
    def params(self):
        """Layer parameters.

        Returns a list of numpy.array variables or expressions that
        parameterize the layer.
        Returns
        -------
        list of numpy.array variables or expressions
            A list of variables that parameterize the layer
        Notes
        -----
        For layers without any parameters, this will return an empty list.
        """
        return []

    @property
    def grads(self):
        """Get layer parameter gradients as calculated from backward()."""
        return []

    @property
    def param_grads(self):
        """Layer parameters and corresponding gradients."""
        return list(zip(self.params, self.grads))

    def __str__(self):
        return self.__class__.__name__
