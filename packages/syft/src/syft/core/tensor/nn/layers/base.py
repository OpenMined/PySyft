# relative
from typing import Union
from ...autodp.phi_tensor import PhiTensor
from ...autodp.gamma_tensor import GammaTensor

from numpy.typing import ArrayLike

from typing_extensions import Self


class Layer:
    """
    Subclassed when implementing new types of layers.

    Each layer can keep track of the layer(s) feeding into it, a
    network's output :class:`Layer` instance can double as a handle to the full
    network.
    """

    first_layer: bool = False

    def forward(self, input: Union[PhiTensor, GammaTensor, ArrayLike], *args, **kwargs):
        raise NotImplementedError

    def backward(
        self, pre_grad: Union[PhiTensor, GammaTensor, ArrayLike], *args, **kwargs
    ):
        raise NotImplementedError

    def connect_to(self, prev_layer: Self) -> None:
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
