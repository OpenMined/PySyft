# stdlib
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union

# third party
import numpy as np

# relative
from ....common.serde.serializable import serializable
from ...autodp.gamma_tensor import GammaTensor
from ...autodp.phi_tensor import PhiTensor
from .base import Layer


@serializable(recursive_serde=True)
class Flatten(Layer):
    __attr_allowlist__ = ("outdim", "last_input_shape", "out_shape", "input_shape")

    def __init__(self, outdim: int = 2) -> None:
        self.outdim = outdim
        if outdim < 1:
            raise ValueError("Dim must be >0, was %i", outdim)

        self.last_input_shape: Optional[Tuple[int, ...]] = None
        self.out_shape: Optional[Tuple[int, ...]] = None

    def connect_to(self, prev_layer: Layer) -> None:
        if len(prev_layer.out_shape) < 2:  # type: ignore
            raise ValueError(
                f"Input layer shape should have more than two dimensions:{prev_layer.out_shape}"  # type: ignore
            )

        to_flatten = np.prod(prev_layer.out_shape[self.outdim - 1 :])  # type: ignore # noqa: E203
        flattened_shape = prev_layer.out_shape[: self.outdim - 1] + (int(to_flatten),)  # type: ignore

        self.input_shape = prev_layer.out_shape  # type: ignore
        self.out_shape = flattened_shape

    def forward(
        self, input: Union[PhiTensor, GammaTensor], *args: Any, **kwargs: Any
    ) -> Union[PhiTensor, GammaTensor]:
        self.last_input_shape = input.shape

        to_flatten = np.prod(self.last_input_shape[self.outdim - 1 :])  # noqa: E203
        flattened_shape = input.shape[: self.outdim - 1] + (int(to_flatten),)
        return input.reshape(flattened_shape)

    def backward(
        self, pre_grad: Union[PhiTensor, GammaTensor], *args: Any, **kwargs: Any
    ) -> Union[PhiTensor, GammaTensor]:
        if self.last_input_shape is None:
            raise ValueError(
                "`self.last_input_shape` cannot be None. \
                Please check the shape of the last input."
            )
        return pre_grad.reshape(self.last_input_shape)
