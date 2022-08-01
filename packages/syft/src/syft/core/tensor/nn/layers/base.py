# stdlib
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# relative
from ...autodp.gamma_tensor import GammaTensor
from ...autodp.phi_tensor import PhiTensor


class Layer:
    """
    Subclassed when implementing new types of layers.

    Each layer can keep track of the layer(s) feeding into it, a
    network's output :class:`Layer` instance can double as a handle to the full
    network.
    """

    first_layer: bool = False

    def forward(
        self,
        input: Union[PhiTensor, GammaTensor],
        *args: Optional[Any],
        **kwargs: Optional[Any]
    ) -> Union[PhiTensor, GammaTensor]:
        raise NotImplementedError

    def backward(
        self,
        pre_grad: Union[PhiTensor, GammaTensor],
        *args: Optional[Any],
        **kwargs: Optional[Any]
    ) -> Optional[Union[PhiTensor, GammaTensor]]:
        raise NotImplementedError

    def connect_to(self, prev_layer: "Layer") -> None:
        raise NotImplementedError

    @property
    def params(self) -> Tuple:
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
        return ()

    @property
    def grads(self) -> Tuple:
        """Get layer parameter gradients as calculated from backward()."""
        return ()

    @property
    def param_grads(self) -> List:
        """Layer parameters and corresponding gradients."""
        return list(zip(self.params, self.grads))

    def __str__(self) -> str:
        return self.__class__.__name__
